"""Read-only acquired-data derivative and feasible Gauss-Newton step audit.

Does not update saved fits, relax acceptance gates or use withheld projections.
"""

from experiments.worm_uncertainty.layout import resource_path

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from experiments.worm_uncertainty.core.cholesky_gaussian import projection
from experiments.worm_uncertainty.core.staged_density import prior_and_bounds
from experiments.worm_uncertainty.data.prepare import sha256


def audit(p, y, angles, uv, reference, cfg):
    mean, sd, low, high = prior_and_bounds(reference, cfg)

    def residual(x):
        pred, jac = projection(x, angles, uv)
        return np.r_[(pred - y) / cfg["noise_sigma"], (x - mean) / sd], np.vstack(
            [jac / cfg["noise_sigma"], np.diag(1 / sd)]
        )

    r, j = residual(p)
    a = j * sd[None, :]
    u, singular, vh = np.linalg.svd(a, full_matrices=False)
    grad = a.T @ r
    # One reproducible mixed direction and steepest descent in prior-SD units.
    rng = np.random.default_rng(20261027)
    directions = [rng.normal(size=len(p)), -grad]
    checks = []
    for name, direction in zip(["random", "negative_gradient"], directions):
        direction /= max(np.linalg.norm(direction), 1e-30)
        analytic = a @ direction
        for h in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            # These are local derivative probes, not accepted feasible updates.
            plus, _ = residual(p + h * sd * direction)
            minus, _ = residual(p - h * sd * direction)
            finite = (plus - minus) / (2 * h)
            checks.append(
                dict(
                    direction=name,
                    h=h,
                    relative_jacobian_error=float(
                        np.linalg.norm(finite - analytic) / max(np.linalg.norm(analytic), 1e-30)
                    ),
                    objective_directional_analytic=float(r @ analytic),
                    objective_directional_fd=float((plus @ plus - minus @ minus) / (4 * h)),
                )
            )
    step = lsq_linear(
        a, -r, bounds=((low - p) / sd, (high - p) / sd), tol=1e-12, max_iter=300, method="bvls"
    )
    before = float(r @ r)
    trials = []
    for alpha in [1.0, 0.1, 0.01, 0.001]:
        candidate = np.clip(p + alpha * sd * step.x, low, high)
        trial, _ = residual(candidate)
        actual = float(trial @ trial)
        linear = r + a @ (alpha * step.x)
        trials.append(
            dict(
                alpha=alpha,
                objective=actual,
                reduction=before - actual,
                predicted_reduction=before - float(linear @ linear),
            )
        )
    return dict(
        objective=before,
        prior_scaled_jacobian_condition=float(singular[0] / singular[-1]),
        singular_min=float(singular[-1]),
        singular_max=float(singular[0]),
        derivative_checks=checks,
        step_solver_success=bool(step.success),
        step_solver_optimality=float(step.optimality),
        step_norm_prior_units=float(np.linalg.norm(step.x)),
        trials=trials,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent", type=Path, default=Path("output/worm_uncertainty/worm_stationarity_v1")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    report = json.loads((args.parent / "report.json").read_text())
    for name, h in report["artifacts_sha256"].items():
        if sha256(args.parent / name) != h:
            raise ValueError("artifact hash mismatch: " + name)
    previous = Path("output/worm_uncertainty/worm_staged_v1/report.json")
    if sha256(previous) != report["parent_report_sha256"]:
        raise ValueError("parent hash mismatch")
    parent = json.loads(previous.read_text())
    for path, key in [
        ("output/worm_uncertainty/worm_shape_pilot_v1/inputs.npz", "input_sha256"),
        ("output/worm_uncertainty/worm_fit_controls_v1/references.npz", "references_sha256"),
    ]:
        if sha256(path) != parent[key]:
            raise ValueError("input hash mismatch")
    data = np.load("output/worm_uncertainty/worm_shape_pilot_v1/inputs.npz")
    ref = np.load("output/worm_uncertainty/worm_fit_controls_v1/references.npz")["16"]
    m = data["measured"]
    angles = data["angles"]
    uv = data["uv"]
    mask = m.reshape(len(angles), len(uv))[:, 0]
    if not np.array_equal(m, np.repeat(mask, len(uv))):
        raise ValueError("complete acquired views required")
    rows = []
    for fit in json.loads((args.parent / "fits.json").read_text()):
        row = dict(
            arm=fit["arm"],
            seed=fit["seed"],
            **audit(
                np.array(fit["parameters"]),
                data["noisy"][m],
                angles[mask],
                uv,
                ref,
                parent["configuration"],
            ),
        )
        rows.append(row)
        print(
            row["arm"],
            row["seed"],
            "condition",
            row["prior_scaled_jacobian_condition"],
            "best trial reduction",
            max(t["reduction"] for t in row["trials"]),
            flush=True,
        )
    args.output.mkdir(parents=True)
    result = dict(
        rows=rows,
        parent_report_sha256=sha256(args.parent / "report.json"),
        source_sha256={
            n: sha256(resource_path(n))
            for n in [
                "worm_numerical_audit.py",
                "cholesky_gaussian.py",
                "staged_density.py",
                "prepare.py",
            ]
        },
        claim="Acquired-only local diagnostics. Trial steps do not replace saved candidates or prove global optimality. No posterior or transfer claim.",
    )
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
