"""Frozen separate-phantom tests of joint centre/density uncertainty."""

from experiments.worm_uncertainty.layout import resource_path

import argparse
import json
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.studies.diagnose_3d import (
    ray_basis,
    voxel_features,
    features,
    optimize,
    invert_precision,
    diagnostic,
)
from experiments.worm_uncertainty.data.prepare import sha256


def acquisition_mask(degrees, start, span):
    return (degrees - start) % 180 < span


def propagated_sd(jacobian, covariance):
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", jacobian, covariance, jacobian), 0))


def evaluate(y, angles, uv, scales, initial, sigma, cfg, mask, points):
    """Fit and compute covariance using acquired data only."""
    pixels = len(uv)
    ray_mask = np.repeat(mask, pixels)
    fit, prior_sd = optimize(
        y[ray_mask], angles[mask], uv, scales, initial, sigma, cfg, cfg["budget_nfev"]
    )
    mean, jac, basis = features(fit.x, scales, angles, uv)
    conditional = invert_precision(
        basis[ray_mask].T @ basis[ray_mask] / sigma**2
        + np.eye(len(scales)) / cfg["density_prior_sd"] ** 2
    )
    joint = invert_precision(jac[ray_mask].T @ jac[ray_mask] / sigma**2 + np.diag(1 / prior_sd**2))
    k = len(scales)
    vb, vd = voxel_features(fit.x[: 3 * k].reshape(k, 3), scales, points)
    vj = np.concatenate(
        [(vd * fit.x[3 * k :][None, :, None]).reshape(len(points), 3 * k), vb], axis=1
    )
    return dict(
        fit=fit,
        ray_mean=mean,
        voxel_mean=vb @ fit.x[3 * k :],
        density_ray_sd=propagated_sd(basis, conditional),
        joint_ray_sd=propagated_sd(jac, joint),
        density_voxel_sd=propagated_sd(vb, conditional),
        joint_voxel_sd=propagated_sd(vj, joint),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--protocol", type=Path, default=resource_path("phantom_stress_protocol_v1.json")
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output exists; use a fresh directory")
    cfg = json.loads(args.protocol.read_text())
    degrees = np.arange(cfg["angles_deg"]["count"]) * cfg["angles_deg"]["step"]
    angles = np.deg2rad(degrees)
    h, w = cfg["detector_shape"]
    u = (np.arange(w) + 0.5) * 2 / w - 1
    v = (np.arange(h) + 0.5) * 2 / h - 1
    uu, vv = np.meshgrid(u, v)
    uv = np.c_[uu.ravel(), vv.ravel()]
    coords = [(np.arange(n) + 0.5) * 2 / n - 1 for n in cfg["voxel_shape"]]
    points = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1).reshape(-1, 3)
    rows = []
    draws = []
    for phantom in cfg["phantoms"]:
        centres = np.array(phantom["centres"])
        scales = np.array(phantom["scales"])
        density = np.array(phantom["densities"])
        for condition in cfg["conditions"]:
            tc, ts, td = centres, scales, density
            if condition == "omitted_component":
                extra = phantom["extra"]
                tc = np.vstack([tc, extra["centre"]])
                ts = np.vstack([ts, extra["scale"]])
                td = np.r_[td, extra["density"]]
            clean = ray_basis(tc, ts, angles, uv)[0] @ td
            truth = voxel_features(tc, ts, points)[0] @ td
            foreground = truth > cfg["foreground_fraction_of_peak"] * truth.max()
            ray_fg = clean > cfg["foreground_fraction_of_peak"] * clean.max()
            sigma = cfg["noise_fraction_full_projection_rms"] * np.sqrt(np.mean(clean**2))
            noise = (
                np.random.default_rng(cfg["noise_seed"]).normal(
                    size=(cfg["noise_repeats"], len(clean))
                )
                * sigma
            )
            for acquisition in cfg["acquisitions"]:
                mask = acquisition_mask(degrees, acquisition["start"], acquisition["span"])
                measured = np.repeat(mask, len(uv))
                withheld = ~measured & ray_fg
                case = dict(phantom=phantom["name"], condition=condition, **acquisition)
                records = []
                for seed in cfg["initialization_seeds"]:
                    initial = np.r_[
                        (
                            centres
                            + np.random.default_rng(seed).normal(
                                0, cfg["initialization_position_sd"], centres.shape
                            )
                        ).ravel(),
                        density * cfg["initialization_density_fraction"],
                    ]
                    for repeat in range(cfg["noise_repeats"]):
                        y = clean + noise[repeat]
                        result = evaluate(y, angles, uv, scales, initial, sigma, cfg, mask, points)
                        fit = result["fit"]
                        record = dict(
                            **case,
                            initialization_seed=seed,
                            noise_repeat=repeat,
                            converged=bool(fit.success),
                            active_bounds=bool(np.any(fit.active_mask)),
                            nfev=fit.nfev,
                            optimality=float(fit.optimality),
                            measured_clean_relative_l2=float(
                                np.linalg.norm((result["ray_mean"] - clean)[measured])
                                / np.linalg.norm(clean[measured])
                            ),
                            methods={},
                        )
                        for method in ["density", "joint"]:
                            metrics = dict(
                                voxel=diagnostic(
                                    result["voxel_mean"],
                                    result[method + "_voxel_sd"],
                                    truth,
                                    foreground,
                                )
                            )
                            if withheld.any():
                                sd = result[method + "_ray_sd"]
                                metrics["withheld_latent"] = diagnostic(
                                    result["ray_mean"], sd, clean, withheld
                                )
                                metrics["withheld_predictive"] = diagnostic(
                                    result["ray_mean"], np.sqrt(sd**2 + sigma**2), y, withheld
                                )
                            record["methods"][method] = metrics
                        records.append(record)
                        draws.append(record)
                row = dict(
                    **case,
                    sigma=float(sigma),
                    fits=len(records),
                    converged=sum(r["converged"] for r in records),
                    active_bounds=sum(r["active_bounds"] for r in records),
                    max_nfev=max(r["nfev"] for r in records),
                    measured_clean_relative_l2=float(
                        np.mean([r["measured_clean_relative_l2"] for r in records])
                    ),
                    methods={},
                )
                for method in ["density", "joint"]:
                    row["methods"][method] = {
                        domain: {
                            metric: float(
                                np.mean([r["methods"][method][domain][metric] for r in records])
                            )
                            for metric in records[0]["methods"][method][domain]
                        }
                        for domain in records[0]["methods"][method]
                    }
                rows.append(row)
                print(json.dumps(row), flush=True)
    args.output.mkdir(parents=True)
    (args.output / "draws.json").write_text(json.dumps(draws, indent=2) + "\n")
    report = dict(
        protocol=cfg,
        protocol_sha256=sha256(args.protocol),
        code_sha256=sha256(__file__),
        dependencies_sha256={
            name: sha256(resource_path(name)) for name in ["diagnose_3d.py", "prepare.py"]
        },
        draws_sha256=sha256(args.output / "draws.json"),
        rows=rows,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
