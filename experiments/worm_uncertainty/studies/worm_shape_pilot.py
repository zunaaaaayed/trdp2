"""Small acquired-data-only worm pilot with full Gaussian shape covariance."""

from experiments.worm_uncertainty.layout import resource_path

import argparse
import json
import time
from pathlib import Path
import numpy as np
import tifffile
from experiments.worm_uncertainty.core.cholesky_gaussian import projection, voxel, fit_control
from experiments.worm_uncertainty.studies.diagnose_3d import ray_basis, invert_precision, diagnostic
from experiments.worm_uncertainty.core.measured_initializer import initialize, perturb
from experiments.worm_uncertainty.studies.stress_3d import acquisition_mask, propagated_sd
from experiments.worm_uncertainty.data.prepare import sha256


def sampling_geometry(cfg):
    spacing = 2 / cfg["source_shape"][2]
    angles = np.deg2rad(np.asarray(cfg["angle_indices"]) * 1.8)
    u = (np.asarray(cfg["detector_indices"]) + 0.5 - cfg["source_shape"][2] / 2) * spacing
    v = (np.asarray(cfg["z_indices"]) - np.mean(cfg["z_indices"])) * spacing
    uu, vv = np.meshgrid(u, v)
    return angles, np.c_[uu.ravel(), vv.ravel()], v


def normalize_acquired(raw, angle_mask):
    scale = float(np.sqrt(np.mean(raw[angle_mask] ** 2)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid acquired-data normalization")
    return raw / scale, scale


def initialize_worm(acquired, angles, uv, v, cfg):
    xy = np.linspace(
        -cfg["initialization_extent"], cfg["initialization_extent"], cfg["initialization_grid_xy"]
    )
    z = np.linspace(v.min(), v.max(), cfg["initialization_grid_z"])
    centres = np.stack(np.meshgrid(xy, xy, z, indexing="ij"), axis=-1).reshape(-1, 3)
    scales = np.tile(
        [cfg["initialization_width"], cfg["initialization_width"], cfg["initialization_width_z"]],
        (len(centres), 1),
    )
    basis = ray_basis(centres, scales, angles, uv)[0]
    reference, info = initialize(
        acquired, angles, uv, cfg, (centres, basis, np.linalg.norm(basis, axis=0))
    )
    k = cfg["components"]
    reference[3 * k : 6 * k] = np.log(scales[info["selected_indices"]]).ravel()
    return reference, info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("../worm_data/worm_004_gt.tiff"))
    p.add_argument("--protocol", type=Path, default=resource_path("worm_shape_protocol_v1.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    cfg = json.loads(args.protocol.read_text())
    if not args.execute:
        print(json.dumps(cfg, indent=2))
        return
    if args.output.exists():
        raise ValueError("output exists")
    if sha256(args.source) != cfg["source_sha256"]:
        raise ValueError("source hash mismatch")
    angles, uv, v = sampling_geometry(cfg)
    mask = acquisition_mask(np.asarray(cfg["angle_indices"]) * 1.8, cfg["start"], cfg["span"])
    measured = np.repeat(mask, len(uv))
    sigma = cfg["noise_sigma"]
    with tifffile.TiffFile(args.source) as tif:
        if list(tif.series[0].shape) != cfg["source_shape"]:
            raise ValueError("TIFF axes mismatch")
        raw = np.stack(
            [
                tif.pages[z].asarray()[np.ix_(cfg["angle_indices"], cfg["detector_indices"])]
                for z in cfg["z_indices"]
            ],
            axis=1,
        )
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite source")
    clean, scale = normalize_acquired(raw, mask)
    clean = clean.ravel()
    y = clean + np.random.default_rng(cfg["noise_seed"]).normal(0, sigma, len(clean))
    reference, info = initialize_worm(y[measured], angles[mask], uv, v, cfg)
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_text(args.protocol.read_text())
    np.savez_compressed(
        args.output / "inputs.npz",
        clean=clean,
        noisy=y,
        angles=angles,
        uv=uv,
        v=v,
        measured=measured,
        reference=reference,
    )
    fits = []
    started = time.monotonic()
    for seed in cfg["initialization_seeds"]:
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
        row = dict(
            seed=seed,
            objective=float(2 * fit.cost),
            converged=bool(fit.success),
            nfev=fit.nfev,
            optimality=float(fit.optimality),
            active_indices=np.flatnonzero(fit.active_mask).tolist(),
            parameters=fit.x.tolist(),
        )
        fits.append(row)
        (args.output / "fits.json").write_text(json.dumps(fits, indent=2) + "\n")
        print(json.dumps({key: val for key, val in row.items() if key != "parameters"}), flush=True)
    converged = sorted([r for r in fits if r["converged"]], key=lambda r: r["objective"])
    chosen = converged[0] if converged else min(fits, key=lambda r: r["objective"])
    gate = dict(passed=False, converged_candidates=len(converged), selected_seed=chosen["seed"])
    if len(converged) >= cfg["stability_gate"]["minimum_converged"]:
        first, second = converged[:2]
        a = projection(np.array(first["parameters"]), angles[mask], uv)[0]
        b = projection(np.array(second["parameters"]), angles[mask], uv)[0]
        gap = abs(first["objective"] - second["objective"]) / max(1, abs(first["objective"]))
        disagreement = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-15))
        gate.update(
            relative_objective_gap=gap,
            relative_prediction_disagreement=disagreement,
            passed=not (first["active_indices"] or second["active_indices"])
            and gap <= cfg["stability_gate"]["relative_objective_gap"]
            and disagreement <= cfg["stability_gate"]["relative_prediction_disagreement"],
        )
    parameters = np.array(chosen["parameters"])
    mean, jac = projection(parameters, angles, uv)
    k = cfg["components"]
    precision = jac[measured].T @ jac[measured] / sigma**2 + np.diag(1 / prior_sd**2)
    covariance = invert_precision(precision)
    density_cov = invert_precision(precision[9 * k :, 9 * k :])
    full_sd = propagated_sd(jac, covariance)
    density_sd = propagated_sd(jac[:, 9 * k :], density_cov)
    selected = (~measured) & (clean > cfg["foreground_threshold"])
    metrics = {}
    for name, sd in [("density", density_sd), ("full_shape", full_sd)]:
        metrics[name] = dict(
            withheld_latent=diagnostic(mean, sd, clean, selected),
            withheld_predictive=diagnostic(mean, np.sqrt(sd**2 + sigma**2), y, selected),
        )
    q = cfg["query_width"]
    xy = (np.arange(q) + 0.5) * 2 / q - 1
    points = np.stack(np.meshgrid(xy, xy, v, indexing="ij"), axis=-1).reshape(-1, 3)
    volume = []
    volume_sd = []
    for offset in range(0, len(points), cfg["query_chunk"]):
        value, derivative = voxel(parameters, points[offset : offset + cfg["query_chunk"]])
        volume.append(value)
        volume_sd.append(propagated_sd(derivative, covariance))
    np.savez_compressed(
        args.output / "arrays.npz",
        prediction=mean,
        full_sd=full_sd,
        density_sd=density_sd,
        volume=np.concatenate(volume).reshape(q, q, len(v)),
        volume_sd=np.concatenate(volume_sd).reshape(q, q, len(v)),
    )
    report = dict(
        kind=cfg["kind"],
        protocol=cfg,
        source_sha256=cfg["source_sha256"],
        normalization=scale,
        initializer=info,
        gate=gate,
        metrics=metrics,
        measured_relative_l2=float(
            np.linalg.norm((mean - clean)[measured]) / np.linalg.norm(clean[measured])
        ),
        withheld_relative_l2=float(
            np.linalg.norm((mean - clean)[~measured]) / np.linalg.norm(clean[~measured])
        ),
        precision_condition_number=float(np.linalg.cond(precision)),
        parameters=10 * k,
        covariance_bytes=covariance.nbytes,
        elapsed_seconds=time.monotonic() - started,
        code_sha256=sha256(__file__),
        protocol_sha256=sha256(args.protocol),
        dependencies_sha256={
            name: sha256(resource_path(name))
            for name in [
                "cholesky_gaussian.py",
                "shape_gaussian.py",
                "measured_initializer.py",
                "rotation_sensitivity.py",
                "stress_3d.py",
                "diagnose_3d.py",
                "prepare.py",
            ]
        },
        artifacts_sha256={
            name: sha256(args.output / name) for name in ["inputs.npz", "arrays.npz", "fits.json"]
        },
        scientific_claim="single developmental projection diagnostic; no voxel truth or calibrated-density claim",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
