"""Known-truth shape-conditioning diagnostic with explicitly oracle geometry priors.

A matched-volume moderate/elongated pair, full/limited angles, clean/noisy data,
and exact/perturbed starts. Not a generalization or uncertainty-calibration test.
"""

from experiments.worm_uncertainty.layout import resource_path

import argparse
import json
import platform
from pathlib import Path

import numpy as np
from experiments.worm_uncertainty.core.cholesky_gaussian import (
    convert_euler,
    geometry,
    projection,
    voxel,
)
from experiments.worm_uncertainty.core.measured_initializer import perturb
from experiments.worm_uncertainty.data.prepare import sha256
from experiments.worm_uncertainty.core.staged_density import prior_and_bounds
from experiments.worm_uncertainty.studies.worm_stationarity import continue_fit


def phantom(elongated):
    centres = np.array(
        [[-0.25, -0.18, -0.04], [0.25, -0.15, 0.04], [-0.2, 0.22, 0.02], [0.2, 0.2, -0.02]]
    )
    widths = np.tile([0.008, 0.8, 0.12] if elongated else [0.08, 0.08, 0.12], (4, 1))
    rotations = np.zeros((4, 3))
    rotations[:, 2] = [0.2, 0.7, -0.4, 1.1]
    return convert_euler(
        np.r_[centres.ravel(), np.log(widths).ravel(), rotations.ravel(), [2.0, 3.0, 1.5, 2.5]]
    )


def relative_error(prediction, truth):
    return float(np.linalg.norm(prediction - truth) / np.linalg.norm(truth))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=resource_path("conditioning_phantom_protocol_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    design = json.loads(args.protocol.read_text())
    if not args.execute:
        print(json.dumps(design, indent=2))
        return
    if args.output.exists():
        raise ValueError("output exists; choose a fresh path")
    cfg = json.loads(
        Path("experiments/worm_uncertainty/protocols/worm_shape_protocol_v1.json").read_text()
    )
    if (
        sha256("experiments/worm_uncertainty/protocols/worm_shape_protocol_v1.json")
        != design["base_protocol_sha256"]
    ):
        raise ValueError("base protocol hash mismatch")
    angles = np.deg2rad(np.arange(0, 180, 7.2))
    uv = np.array([[u, z] for z in np.linspace(-0.16, 0.16, 9) for u in np.linspace(-1, 1, 129)])
    points = np.stack(
        np.meshgrid(
            np.linspace(-1, 1, 101),
            np.linspace(-1, 1, 101),
            np.linspace(-0.16, 0.16, 17),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_text(args.protocol.read_text())
    source_names = [
        "conditioning_phantom.py",
        "cholesky_gaussian.py",
        "shape_gaussian.py",
        "worm_stationarity.py",
        "staged_density.py",
        "measured_initializer.py",
        "prepare.py",
        "rotation_sensitivity.py",
        "stress_3d.py",
    ]
    provenance = dict(
        protocol=design,
        base_configuration=cfg,
        python=platform.python_version(),
        numpy=np.__version__,
        source_sha256={n: sha256(resource_path(n)) for n in source_names},
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    fits = []
    arrays = dict(angles=angles, uv=uv, voxel_points=points)
    for shape in ["moderate", "elongated"]:
        truth = phantom(shape == "elongated")
        clean = projection(truth, angles, uv)[0].reshape(len(angles), len(uv))
        true_voxels = voxel(truth, points)[0]
        arrays[shape + "_truth"] = truth
        arrays[shape + "_clean"] = clean
        for span in design["spans"]:
            mask = np.rad2deg(angles) < span
            acquired = clean[mask].ravel()
            sigma = design["relative_noise_sd"] * np.sqrt(np.mean(acquired**2))
            local = dict(cfg, components=4, noise_sigma=float(sigma))
            noise = np.random.default_rng(design["noise_seed"]).normal(size=len(acquired)) * sigma
            for noisy in [False, True]:
                y = acquired + noise if noisy else acquired.copy()
                name = f"{shape}_{span}_{int(noisy)}"
                arrays[name + "_y"] = y
                for seed in design["start_seeds"]:
                    initial = perturb(truth, seed, local)
                    fit = continue_fit(initial, y, angles[mask], uv, truth, local, design)
                    p = np.array(fit["parameters"])
                    predicted = projection(p, angles, uv)[0].reshape(clean.shape)
                    fit.update(
                        shape=shape,
                        span=span,
                        noisy=noisy,
                        seed=seed,
                        sigma=float(sigma),
                        acquired_clean_relative_l2=relative_error(predicted[mask], clean[mask]),
                        withheld_relative_l2=relative_error(predicted[~mask], clean[~mask])
                        if (~mask).any()
                        else None,
                        voxel_relative_l2=relative_error(voxel(p, points)[0], true_voxels),
                        prior_is_oracle_geometry=True,
                    )
                    mean, sd, _, _ = prior_and_bounds(truth, local)
                    truth_r = (projection(truth, angles[mask], uv)[0] - y) / sigma
                    fit["truth_objective"] = float(
                        truth_r @ truth_r + np.sum(((truth - mean) / sd) ** 2)
                    )
                    fits.append(fit)
                    (args.output / "fits.json").write_text(json.dumps(fits, indent=2) + "\n")
                    print(
                        json.dumps(
                            {
                                k: fit[k]
                                for k in [
                                    "shape",
                                    "span",
                                    "noisy",
                                    "seed",
                                    "converged",
                                    "status",
                                    "nfev",
                                    "objective",
                                    "voxel_relative_l2",
                                    "withheld_relative_l2",
                                ]
                            }
                        ),
                        flush=True,
                    )
    np.savez_compressed(args.output / "inputs.npz", **arrays)
    rows = [
        {k: v for k, v in f.items() if k not in ["parameters", "before", "after"]}
        | dict(projected_gradient_inf=f["after"]["projected_gradient_inf"])
        for f in fits
    ]
    report = dict(
        **provenance,
        rows=rows,
        artifacts_sha256={
            n: sha256(args.output / n)
            for n in ["fits.json", "inputs.npz", "protocol.json", "provenance.json"]
        },
        claim="Oracle geometry prior and truth-based starts; finite sampled voxel region. Density prior mean remains zero, so noiseless truth need not minimize the regularized objective. No calibration/generalization claim.",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
