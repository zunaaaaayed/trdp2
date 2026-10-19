"""Add fixed extra starts only to cases rejected by the acquired-data stability gate."""

import argparse
import json
from pathlib import Path
import numpy as np
from cholesky_gaussian import convert_euler, projection, voxel, fit_control
from shape_gaussian import uncertainty
from measured_initializer import initialize, perturb
from stress_3d import acquisition_mask
from diagnose_3d import diagnostic
from prepare import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).with_name("measured_fallback_protocol_v1.json"),
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    policy = json.loads(args.policy.read_text())
    report = json.loads((args.input / "report.json").read_text())
    cfg = report["protocol"]
    audit = json.loads((args.input / "stability_gate_audit.json").read_text())
    if (
        sha256(args.input / "report.json") != policy["parent_report_sha256"]
        or sha256(args.input / "stability_gate_audit.json") != policy["parent_audit_sha256"]
    ):
        raise ValueError("parent provenance changed")
    if (
        sha256(args.input / "draws.json") != report["draws_sha256"]
        or sha256(args.input / "arrays.npz") != report["arrays_sha256"]
    ):
        raise ValueError("parent artifact changed")
    if cfg["budget_nfev"] != policy["budget_nfev"] or set(cfg["initialization_seeds"]) & set(
        policy["extra_seeds"]
    ):
        raise ValueError("fallback must keep budget and add distinct starts")
    if (
        len(cfg["initialization_seeds"]) + len(policy["extra_seeds"])
        > policy["maximum_total_starts"]
    ):
        raise ValueError("start limit exceeded")
    draws = json.loads((args.input / "draws.json").read_text())
    arrays = dict(np.load(args.input / "arrays.npz"))
    additional = []
    degrees = np.arange(cfg["angles_deg"]["count"]) * cfg["angles_deg"]["step"]
    angles = np.deg2rad(degrees)
    h, w = cfg["detector_shape"]
    uu, vv = np.meshgrid((np.arange(w) + 0.5) * 2 / w - 1, (np.arange(h) + 0.5) * 2 / h - 1)
    uv = np.c_[uu.ravel(), vv.ravel()]
    coords = [(np.arange(n) + 0.5) * 2 / n - 1 for n in cfg["voxel_shape"]]
    points = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1).reshape(-1, 3)
    sigma = cfg["noise_sigma"]
    for failure in audit["records"]:
        if failure["passed"]:
            continue
        case = next(c for c in cfg["cases"] if c["name"] == failure["case"])
        phantom = next(p for p in cfg["phantoms"] if case["name"].startswith(p["name"] + "_"))
        # Regenerate the frozen synthetic input, never pass truth to fitting.
        target = convert_euler(
            np.r_[
                np.array(phantom["centres"]).ravel(),
                np.log(phantom["scales"]).ravel(),
                np.array(phantom["rotations"]).ravel(),
                phantom["densities"],
            ]
        )
        clean = projection(target, angles, uv)[0]
        truth = voxel(target, points)[0]
        noise = (
            np.random.default_rng(cfg["noise_seed"]).normal(size=(cfg["noise_repeats"], len(clean)))
            * sigma
        )
        repeat = failure["noise_repeat"]
        y = clean + noise[repeat]
        mask = acquisition_mask(degrees, case["start"], case["span"])
        measured = np.repeat(mask, len(uv))
        reference, _ = initialize(y[measured], angles[mask], uv, cfg)
        previous = next(
            r for r in draws if r["case"] == case["name"] and r["noise_repeat"] == repeat
        )
        np.testing.assert_array_equal(reference, previous["prior_reference"])
        fg = truth > cfg["foreground_fraction_of_peak"] * truth.max()
        withheld = ~measured & (clean > cfg["foreground_fraction_of_peak"] * clean.max())
        for seed in policy["extra_seeds"]:
            fit, prior_sd = fit_control(
                y[measured],
                angles[mask],
                uv,
                perturb(reference, seed, cfg),
                reference,
                sigma,
                cfg,
                False,
            )
            mean, jac = projection(fit.x, angles, uv)
            vmean, vjac = voxel(fit.x, points)
            ray_sd, v_sd = uncertainty(jac, vjac, measured, sigma, prior_sd, cfg["components"])[
                "rotations"
            ]
            record = dict(
                case=case["name"],
                noise_repeat=repeat,
                initialization_seed=seed,
                objective=float(2 * fit.cost),
                converged=bool(fit.success),
                nfev=fit.nfev,
                active_indices=np.flatnonzero(fit.active_mask).tolist(),
                parameters=fit.x.tolist(),
                prior_reference=reference.tolist(),
                metrics=dict(
                    voxel=diagnostic(vmean, v_sd, truth, fg),
                    withheld_latent=diagnostic(mean, ray_sd, clean, withheld),
                    withheld_predictive=diagnostic(
                        mean, np.sqrt(ray_sd**2 + sigma**2), y, withheld
                    ),
                ),
            )
            additional.append(record)
            if repeat == 0:
                arrays[f"{case['name']}_mean_{seed}"] = vmean.reshape(cfg["voxel_shape"])
                arrays[f"{case['name']}_sd_{seed}"] = v_sd.reshape(cfg["voxel_shape"])
        print(case["name"], repeat, "added", len(policy["extra_seeds"]), "starts", flush=True)
    combined = draws + additional
    rows = []
    for case in cfg["cases"]:
        records = [r for r in combined if r["case"] == case["name"]]
        rows.append(
            dict(
                case=case["name"],
                fits=len(records),
                converged=sum(r["converged"] for r in records),
                active_bounds=sum(bool(r["active_indices"]) for r in records),
                max_nfev=max(r["nfev"] for r in records),
            )
        )
    args.output.mkdir(parents=True)
    (args.output / "draws.json").write_text(json.dumps(combined, indent=2) + "\n")
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    expanded = dict(cfg, fallback_policy=policy)
    result = dict(
        protocol=expanded,
        policy_sha256=sha256(args.policy),
        parent_protocol_sha256=report["protocol_sha256"],
        code_sha256=sha256(__file__),
        dependencies_sha256={
            name: sha256(Path(__file__).with_name(name))
            for name in [
                "measured_initializer.py",
                "cholesky_gaussian.py",
                "shape_gaussian.py",
                "rotation_sensitivity.py",
                "stress_3d.py",
                "diagnose_3d.py",
                "prepare.py",
            ]
        },
        draws_sha256=sha256(args.output / "draws.json"),
        arrays_sha256=sha256(args.output / "arrays.npz"),
        additional_fits=len(additional),
        rows=rows,
    )
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
