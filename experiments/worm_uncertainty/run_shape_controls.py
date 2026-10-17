"""Paired component restoration followed by nested shape-uncertainty controls."""

import argparse
import json
from pathlib import Path
import numpy as np
from prepare import sha256
from diagnose_3d import ray_basis, voxel_features, diagnostic
from stress_3d import acquisition_mask, evaluate
from shape_gaussian import projection, voxel, fit_shape, uncertainty


def summarize(case, records, sigma):
    row = dict(
        **case,
        sigma=float(sigma),
        fits=len(records),
        converged=sum(r["converged"] for r in records),
        active_bounds=sum(r["active_bounds"] for r in records),
        max_nfev=max(r["nfev"] for r in records),
        methods={},
    )
    for method in records[0]["methods"]:
        row["methods"][method] = {
            domain: {
                metric: float(np.mean([r["methods"][method][domain][metric] for r in records]))
                for metric in records[0]["methods"][method][domain]
            }
            for domain in records[0]["methods"][method]
        }
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("shape_protocol_v1.json")
    )
    p.add_argument("--parent", type=Path, default=Path("output/worm_uncertainty/phantom_stress_v1"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    cfg = json.loads(args.protocol.read_text())
    parent = json.loads((args.parent / "report.json").read_text())
    if (
        parent["protocol_sha256"] != cfg["parent_protocol_sha256"]
        or sha256(args.parent / "draws.json") != parent["draws_sha256"]
    ):
        raise ValueError("parent provenance mismatch")
    degrees = np.arange(cfg["angles_deg"]["count"]) * cfg["angles_deg"]["step"]
    angles = np.deg2rad(degrees)
    h, w = cfg["detector_shape"]
    uu, vv = np.meshgrid((np.arange(w) + 0.5) * 2 / w - 1, (np.arange(h) + 0.5) * 2 / h - 1)
    uv = np.c_[uu.ravel(), vv.ravel()]
    coords = [(np.arange(n) + 0.5) * 2 / n - 1 for n in cfg["voxel_shape"]]
    points = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1).reshape(-1, 3)
    rows = []
    draws = []
    arrays = {}
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_text(args.protocol.read_text())
    for phase in ["restored_component", "learned_shape"]:
        for phantom in cfg["phantoms"]:
            extra = phantom["extra"]
            centres = np.vstack([phantom["centres"], extra["centre"]])
            scales = np.vstack([phantom["scales"], extra["scale"]])
            density = np.r_[phantom["densities"], extra["density"]]
            k = len(density)
            if phase == "restored_component":
                clean = ray_basis(centres, scales, angles, uv)[0] @ density
                truth = voxel_features(centres, scales, points)[0] @ density
            else:
                target = np.r_[
                    centres.ravel(),
                    np.log(scales).ravel(),
                    np.array(cfg["truth_rotations"]).ravel(),
                    density,
                ]
                clean = projection(target, angles, uv)[0]
                truth = voxel(target, points)[0]
            fg = truth > cfg["foreground_fraction_of_peak"] * truth.max()
            ray_fg = clean > cfg["foreground_fraction_of_peak"] * clean.max()
            sigma = cfg["noise_fraction_full_projection_rms"] * np.sqrt(np.mean(clean**2))
            noise = (
                np.random.default_rng(cfg["noise_seed"]).normal(
                    size=(cfg["noise_repeats"], len(clean))
                )
                * sigma
            )
            acquisitions = (
                cfg["acquisitions"] if phase == "restored_component" else cfg["shape_acquisitions"]
            )
            for acquisition in acquisitions:
                mask = acquisition_mask(degrees, acquisition["start"], acquisition["span"])
                measured = np.repeat(mask, len(uv))
                withheld = ~measured & ray_fg
                case = dict(phase=phase, phantom=phantom["name"], **acquisition)
                records = []
                if phase == "restored_component":
                    old = next(
                        r
                        for r in parent["rows"]
                        if r["phantom"] == phantom["name"]
                        and r["condition"] == "omitted_component"
                        and r["span"] == acquisition["span"]
                        and r["start"] == acquisition["start"]
                    )
                    if not np.isclose(old["sigma"], sigma, rtol=0, atol=1e-15):
                        raise ValueError("repair must share original truth/noise scale")
                for seed in cfg["initialization_seeds"]:
                    rng = np.random.default_rng(seed)
                    init_centres = centres + rng.normal(
                        0, cfg["initialization_position_sd"], centres.shape
                    )
                    if phase == "restored_component":
                        initial = np.r_[
                            init_centres.ravel(), density * cfg["initialization_density_fraction"]
                        ]
                    else:
                        initial = np.r_[
                            init_centres.ravel(),
                            (
                                np.log(scales)
                                + rng.normal(0, cfg["initialization_log_width_sd"], scales.shape)
                            ).ravel(),
                            (
                                np.array(cfg["truth_rotations"])
                                + rng.normal(0, cfg["initialization_rotation_sd"], scales.shape)
                            ).ravel(),
                            density * cfg["initialization_density_fraction"],
                        ]
                    for repeat in range(cfg["noise_repeats"]):
                        y = clean + noise[repeat]
                        if phase == "restored_component":
                            result = evaluate(
                                y, angles, uv, scales, initial, sigma, cfg, mask, points
                            )
                            fit = result["fit"]
                            mean = result["ray_mean"]
                            vmean = result["voxel_mean"]
                            deviations = {
                                m: (result[m + "_ray_sd"], result[m + "_voxel_sd"])
                                for m in ["density", "joint"]
                            }
                        else:
                            fit, prior_sd = fit_shape(
                                y[measured], angles[mask], uv, initial, sigma, cfg
                            )
                            mean, jac = projection(fit.x, angles, uv)
                            vmean, vjac = voxel(fit.x, points)
                            deviations = uncertainty(jac, vjac, measured, sigma, prior_sd, k)
                        record = dict(
                            **case,
                            initialization_seed=seed,
                            noise_repeat=repeat,
                            converged=bool(fit.success),
                            active_bounds=bool(np.any(fit.active_mask)),
                            nfev=fit.nfev,
                            optimality=float(fit.optimality),
                            parameters=fit.x.tolist(),
                            methods={},
                        )
                        for method, (ray_sd, v_sd) in deviations.items():
                            metrics = dict(voxel=diagnostic(vmean, v_sd, truth, fg))
                            if withheld.any():
                                metrics["withheld_latent"] = diagnostic(
                                    mean, ray_sd, clean, withheld
                                )
                                metrics["withheld_predictive"] = diagnostic(
                                    mean, np.sqrt(ray_sd**2 + sigma**2), y, withheld
                                )
                            record["methods"][method] = metrics
                        records.append(record)
                        draws.append(record)
                        if repeat == 0 and seed == cfg["initialization_seeds"][0]:
                            key = f"{phase}_{phantom['name']}_{acquisition['span']}_{acquisition['start']}"
                            arrays[key + "_truth"] = truth.reshape(cfg["voxel_shape"])
                            arrays[key + "_mean"] = vmean.reshape(cfg["voxel_shape"])
                row = summarize(case, records, sigma)
                rows.append(row)
                (args.output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(json.dumps(row), flush=True)
    (args.output / "draws.json").write_text(json.dumps(draws, indent=2) + "\n")
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    report = dict(
        protocol=cfg,
        protocol_sha256=sha256(args.protocol),
        parent_report_sha256=sha256(args.parent / "report.json"),
        code_sha256=sha256(__file__),
        dependencies_sha256={
            name: sha256(Path(__file__).with_name(name))
            for name in ["shape_gaussian.py", "stress_3d.py", "diagnose_3d.py", "prepare.py"]
        },
        draws_sha256=sha256(args.output / "draws.json"),
        arrays_sha256=sha256(args.output / "arrays.npz"),
        rows=rows,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
