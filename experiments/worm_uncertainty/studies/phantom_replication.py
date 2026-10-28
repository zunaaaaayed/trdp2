"""Paired noise/offset/density-prior replication, without changing geometry priors.

All candidates retained. Exact-start results are the predeclared primary series;
perturbed starts diagnose agreement and are not extra independent observations.
"""

from experiments.worm_uncertainty.layout import resource_path

import argparse
import json
from pathlib import Path

import numpy as np
from experiments.worm_uncertainty.core.cholesky_gaussian import projection
from experiments.worm_uncertainty.studies.conditioning_phantom import phantom, relative_error
from experiments.worm_uncertainty.core.gaussian_l2 import relative_l2
from experiments.worm_uncertainty.core.measured_initializer import perturb
from experiments.worm_uncertainty.data.prepare import sha256
from experiments.worm_uncertainty.studies.worm_stationarity import continue_fit


def acquisition_mask(degrees, offset, span=120):
    return np.mod(np.asarray(degrees) - offset, 180) < span - 1e-9


def observation(clean, mask, noise_seed, relative_noise_sd):
    sigma = float(relative_noise_sd * np.sqrt(np.mean(clean[mask] ** 2)))
    if noise_seed is None:
        return clean[mask].ravel().copy(), sigma
    # Whole-stack noise fixes ray identity across overlapping angular windows.
    noise = np.random.default_rng(noise_seed).normal(size=clean.shape)
    return (clean[mask] + sigma * noise[mask]).ravel(), sigma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=resource_path("phantom_replication_protocol_v1.json"),
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
    base = Path("experiments/worm_uncertainty/protocols/worm_shape_protocol_v1.json")
    if sha256(base) != design["base_protocol_sha256"]:
        raise ValueError("base protocol mismatch")
    cfg = json.loads(base.read_text())
    degrees = np.arange(25) * 7.2
    angles = np.deg2rad(degrees)
    uv = np.array([[u, z] for z in np.linspace(-0.16, 0.16, 9) for u in np.linspace(-1, 1, 129)])
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_text(args.protocol.read_text())
    sources = [
        "phantom_replication.py",
        "conditioning_phantom.py",
        "gaussian_l2.py",
        "worm_stationarity.py",
        "staged_density.py",
        "cholesky_gaussian.py",
        "shape_gaussian.py",
        "measured_initializer.py",
        "prepare.py",
        "rotation_sensitivity.py",
        "stress_3d.py",
    ]
    provenance = dict(
        protocol=design,
        configuration=cfg,
        source_sha256={n: sha256(resource_path(n)) for n in sources},
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    prepared = []
    arrays = dict(angles=angles, uv=uv)
    for shape in ["moderate", "elongated"]:
        truth = phantom(shape == "elongated")
        clean = projection(truth, angles, uv)[0].reshape(25, len(uv))
        arrays[shape + "_truth"] = truth
        arrays[shape + "_clean"] = clean
        for offset in design["offsets_degrees"]:
            mask = acquisition_mask(degrees, offset)
            if mask.sum() != 17:
                raise ValueError("expected 17 acquired angles")
            for noise_seed in [None] + design["noise_seeds"]:
                y, sigma = observation(clean, mask, noise_seed, design["relative_noise_sd"])
                name = f"{shape}_{offset}_{noise_seed}"
                arrays[name + "_y"] = y
                arrays[name + "_mask"] = mask
                prepared.append((shape, truth, clean, offset, mask, noise_seed, y, sigma, name))
    np.savez_compressed(args.output / "inputs.npz", **arrays)
    fits = []
    for shape, truth, clean, offset, mask, noise_seed, y, sigma, name in prepared:
        for tau in design["density_prior_sds"]:
            local = dict(cfg, components=4, noise_sigma=sigma, density_prior_sd=tau)
            for seed in design["start_seeds"]:
                initial = perturb(truth, seed, local)
                fit = continue_fit(initial, y, angles[mask], uv, truth, local, design)
                prediction = projection(np.array(fit["parameters"]), angles, uv)[0].reshape(
                    clean.shape
                )
                fit.update(
                    shape=shape,
                    offset=offset,
                    noise_seed=noise_seed,
                    seed=seed,
                    tau=tau,
                    input_key=name,
                    sigma=sigma,
                    exact_volume_relative_l2=relative_l2(np.array(fit["parameters"]), truth),
                    acquired_clean_relative_l2=relative_error(prediction[mask], clean[mask]),
                    withheld_relative_l2=relative_error(prediction[~mask], clean[~mask]),
                )
                fits.append(fit)
                (args.output / "fits.json").write_text(json.dumps(fits, indent=2) + "\n")
                print(
                    json.dumps(
                        dict(
                            completed=len(fits),
                            **{
                                k: fit[k]
                                for k in [
                                    "shape",
                                    "offset",
                                    "noise_seed",
                                    "seed",
                                    "tau",
                                    "converged",
                                    "status",
                                    "exact_volume_relative_l2",
                                    "withheld_relative_l2",
                                ]
                            },
                        )
                    ),
                    flush=True,
                )
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
            for n in ["inputs.npz", "fits.json", "protocol.json", "provenance.json"]
        },
        claim="Oracle geometry prior, two designed shapes, five noise draws per offset. Prior arms share identical observations and starts. No hyperparameter selection, posterior coverage or independent anatomy claim.",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
