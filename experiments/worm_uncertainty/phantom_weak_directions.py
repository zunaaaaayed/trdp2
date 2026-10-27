"""Probe least-sensitive acquired-data directions without modifying saved fits.

Directions use prior-SD parameter units and data Jacobians only. The larger
joint subspace can have smaller singular values by construction; this diagnostic
measures actual finite-step density/projection changes, not posterior coverage.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cholesky_gaussian import projection
from gaussian_l2 import relative_l2
from phantom_replication import acquisition_mask
from prepare import sha256
from staged_density import prior_and_bounds


def probes(p, angles, uv, mask, reference, cfg):
    _, sd, lower, upper = prior_and_bounds(reference, cfg)
    k = len(p) // 10
    baseline, jac = projection(p, angles, uv)
    raymask = np.repeat(mask, len(uv))
    results = []
    for mode, columns in [("density", np.arange(9 * k, 10 * k)), ("joint", np.arange(10 * k))]:
        whitened = jac[raymask][:, columns] * sd[columns] / cfg["noise_sigma"]
        _, singular, vh = np.linalg.svd(whitened, full_matrices=False)
        direction = np.zeros(len(p))
        direction[columns] = vh[-1]
        delta = sd * direction
        for sign in [-1, 1]:
            signed = sign * delta
            limits = []
            for i in range(len(p)):
                if signed[i] > 0 and np.isfinite(upper[i]):
                    limits.append((upper[i] - p[i]) / signed[i])
                elif signed[i] < 0 and np.isfinite(lower[i]):
                    limits.append((lower[i] - p[i]) / signed[i])
            feasible = max(0.0, min(limits, default=np.inf))
            for requested in [0.001, 0.01, 0.1]:
                step = min(requested, 0.8 * feasible)
                if step <= 1e-12:
                    results.append(
                        dict(mode=mode, sign=sign, requested_step=requested, skipped=True)
                    )
                    continue
                candidate = p + step * signed
                predicted = projection(candidate, angles, uv)[0]
                difference = predicted - baseline
                linear = jac @ (step * signed)
                acquired = difference[raymask]
                results.append(
                    dict(
                        mode=mode,
                        sign=sign,
                        requested_step=requested,
                        step=float(step),
                        skipped=False,
                        min_data_singular=float(singular[-1]),
                        max_data_singular=float(singular[0]),
                        prior_unit_density_fraction=float(np.linalg.norm(direction[9 * k :]) ** 2),
                        acquired_change_sigma_rms=float(
                            np.sqrt(np.mean(acquired**2)) / cfg["noise_sigma"]
                        ),
                        acquired_relative_change=float(
                            np.linalg.norm(acquired) / np.linalg.norm(baseline[raymask])
                        ),
                        withheld_relative_change=float(
                            np.linalg.norm(difference[~raymask])
                            / np.linalg.norm(baseline[~raymask])
                        ),
                        density_relative_change=relative_l2(candidate, p),
                        linearization_relative_error=float(
                            np.linalg.norm(acquired - linear[raymask])
                            / max(np.linalg.norm(acquired), 1e-30)
                        ),
                    )
                )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent", type=Path, default=Path("output/worm_uncertainty/phantom_replication_v1")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    report = json.loads((args.parent / "report.json").read_text())
    for name, digest in report["artifacts_sha256"].items():
        if sha256(args.parent / name) != digest:
            raise ValueError("artifact mismatch: " + name)
    fits = json.loads((args.parent / "fits.json").read_text())
    selected = [
        f for f in fits if f["seed"] == 0 and f["tau"] == 10 and f["noise_seed"] is not None
    ]
    if len(selected) != 30:
        raise ValueError("expected all 30 primary noisy fits at tau 10")
    data = np.load(args.parent / "inputs.npz")
    angles, uv = data["angles"], data["uv"]
    rows = []
    for f in selected:
        cfg = dict(report["configuration"], noise_sigma=f["sigma"], density_prior_sd=10.0)
        mask = acquisition_mask(np.rad2deg(angles), f["offset"])
        if not np.array_equal(mask, data[f["input_key"] + "_mask"]):
            raise ValueError("mask mismatch")
        result = probes(
            np.array(f["parameters"]), angles, uv, mask, data[f["shape"] + "_truth"], cfg
        )
        rows.append(
            dict(shape=f["shape"], offset=f["offset"], noise_seed=f["noise_seed"], probes=result)
        )
        print(f["shape"], f["offset"], f["noise_seed"], "complete", flush=True)
    args.output.mkdir(parents=True)
    result = dict(
        rows=rows,
        parent_report_sha256=sha256(args.parent / "report.json"),
        source_sha256={
            n: sha256(Path(__file__).with_name(n))
            for n in [
                "phantom_weak_directions.py",
                "gaussian_l2.py",
                "cholesky_gaussian.py",
                "staged_density.py",
                "phantom_replication.py",
            ]
        },
        protocol=dict(
            primary="All 30 noisy exact-start fits, density prior SD 10; no stationarity filtering",
            modes=["density", "joint"],
            radii=[0.001, 0.01, 0.1],
            signs=[-1, 1],
            bound_fraction=0.8,
            direction="Smallest right singular vector of acquired data Jacobian, noise-whitened and scaled by prior SD. No prior curvature in this SVD.",
        ),
        claim="Local conditional sensitivity, not posterior sampling, calibration, global identifiability or a proposed regularizer.",
    )
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
