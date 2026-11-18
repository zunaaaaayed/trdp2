"""Fresh-phantom assessment with acquired-data-only starts and shared priors."""

from experiments.worm_uncertainty.layout import resource_path

import argparse
import json
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.core.cholesky_gaussian import (
    convert_euler,
    projection,
    voxel,
    fit_control,
)
from experiments.worm_uncertainty.core.shape_gaussian import uncertainty
from experiments.worm_uncertainty.core.measured_initializer import dictionary, initialize, perturb
from experiments.worm_uncertainty.studies.stress_3d import acquisition_mask
from experiments.worm_uncertainty.studies.diagnose_3d import diagnostic
from experiments.worm_uncertainty.data.prepare import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=resource_path("measured_init_protocol_v1.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    cfg = json.loads(args.protocol.read_text())
    degrees = np.arange(cfg["angles_deg"]["count"]) * cfg["angles_deg"]["step"]
    angles = np.deg2rad(degrees)
    h, w = cfg["detector_shape"]
    uu, vv = np.meshgrid((np.arange(w) + 0.5) * 2 / w - 1, (np.arange(h) + 0.5) * 2 / h - 1)
    uv = np.c_[uu.ravel(), vv.ravel()]
    coords = [(np.arange(n) + 0.5) * 2 / n - 1 for n in cfg["voxel_shape"]]
    points = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1).reshape(-1, 3)
    sigma = cfg["noise_sigma"]
    rows = []
    draws = []
    arrays = {}
    cases = []
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_text(args.protocol.read_text())
    for phantom in cfg["phantoms"]:
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
        fg = truth > cfg["foreground_fraction_of_peak"] * truth.max()
        ray_fg = clean > cfg["foreground_fraction_of_peak"] * clean.max()
        noise = (
            np.random.default_rng(cfg["noise_seed"]).normal(size=(cfg["noise_repeats"], len(clean)))
            * sigma
        )
        for acquisition in cfg["acquisitions"]:
            name = f"{phantom['name']}_{acquisition['span']}_{acquisition['start']}"
            cases.append(dict(name=name, **acquisition))
            mask = acquisition_mask(degrees, acquisition["start"], acquisition["span"])
            measured = np.repeat(mask, len(uv))
            withheld = ~measured & ray_fg
            prepared = dictionary(angles[mask], uv, cfg)
            records = []
            initializer_records = []
            for repeat in range(cfg["noise_repeats"]):
                y = clean + noise[repeat]
                # Only acquired y and acquired angles enter the initializer/fit.
                reference, info = initialize(y[measured], angles[mask], uv, cfg, prepared)
                initializer_records.append(info)
                for seed in cfg["initialization_seeds"]:
                    initial = perturb(reference, seed, cfg)
                    fit, prior_sd = fit_control(
                        y[measured], angles[mask], uv, initial, reference, sigma, cfg, False
                    )
                    mean, jac = projection(fit.x, angles, uv)
                    vmean, vjac = voxel(fit.x, points)
                    ray_sd, v_sd = uncertainty(
                        jac, vjac, measured, sigma, prior_sd, cfg["components"]
                    )["rotations"]
                    record = dict(
                        case=name,
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
                    records.append(record)
                    draws.append(record)
                    if repeat == 0:
                        arrays[f"{name}_truth"] = truth.reshape(cfg["voxel_shape"])
                        arrays[f"{name}_mean_{seed}"] = vmean.reshape(cfg["voxel_shape"])
                        arrays[f"{name}_sd_{seed}"] = v_sd.reshape(cfg["voxel_shape"])
            row = dict(
                case=name,
                fits=len(records),
                converged=sum(r["converged"] for r in records),
                active_bounds=sum(bool(r["active_indices"]) for r in records),
                max_nfev=max(r["nfev"] for r in records),
                initializers=initializer_records,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            (args.output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "draws.json").write_text(json.dumps(draws, indent=2) + "\n")
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    expanded = dict(cfg, cases=cases)
    report = dict(
        protocol=expanded,
        protocol_sha256=sha256(args.protocol),
        code_sha256=sha256(__file__),
        dependencies_sha256={
            name: sha256(resource_path(name))
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
        rows=rows,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
