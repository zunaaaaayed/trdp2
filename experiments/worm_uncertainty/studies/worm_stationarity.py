"""Acquired-only stationarity audit and fixed-budget continuation of saved fits."""

from experiments.worm_uncertainty.layout import resource_path

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from experiments.worm_uncertainty.core.cholesky_gaussian import geometry, projection
from experiments.worm_uncertainty.data.prepare import sha256
from experiments.worm_uncertainty.core.staged_density import prior_and_bounds
from experiments.worm_uncertainty.studies.worm_fit_controls import stability


def diagnostics(p, y, angles, uv, reference, cfg):
    mean, sd, lower, upper = prior_and_bounds(reference, cfg)
    prediction, jac = projection(p, angles, uv)
    residual = (prediction - y) / cfg["noise_sigma"]
    prior = (p - mean) / sd
    gradient = jac.T @ residual / cfg["noise_sigma"] + prior / sd
    # Feasible projected gradient for half the shared sum-of-squares objective.
    projected = p - np.clip(p - gradient, lower, upper)
    centres, covariance, _, density = geometry(p)
    widths = np.sqrt(np.linalg.eigvalsh(covariance))
    return dict(
        objective=float(residual @ residual + prior @ prior),
        projected_gradient_inf=float(np.linalg.norm(projected, ord=np.inf)),
        raw_gradient_inf=float(np.linalg.norm(gradient, ord=np.inf)),
        acquired_residual_rms_sigma=float(np.sqrt(np.mean(residual**2))),
        residual_rms_sigma_by_angle=np.sqrt(
            np.mean(residual.reshape(len(angles), len(uv)) ** 2, axis=1)
        ).tolist(),
        principal_widths=widths.tolist(),
        max_axis_ratio=float(np.max(widths[:, -1] / widths[:, 0])),
        centres=centres.tolist(),
        densities=density.tolist(),
    )


def continue_fit(p, y, angles, uv, reference, cfg, design):
    mean, sd, lower, upper = prior_and_bounds(reference, cfg)
    cache = {}
    calls = 0

    def values(x):
        nonlocal calls
        if "x" not in cache or not np.array_equal(x, cache["x"]):
            cache["x"] = x.copy()
            cache["v"] = projection(x, angles, uv)
            calls += 1
        return cache["v"]

    def residual(x):
        return np.r_[(values(x)[0] - y) / cfg["noise_sigma"], (x - mean) / sd]

    def jacobian(x):
        return np.vstack([values(x)[1] / cfg["noise_sigma"], np.diag(1 / sd)])

    before = diagnostics(p, y, angles, uv, reference, cfg)
    started = time.monotonic()
    result = least_squares(
        residual,
        p,
        jac=jacobian,
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=design["additional_nfev"],
        ftol=None,
        xtol=design["xtol"],
        gtol=design["gtol"],
    )
    after = diagnostics(result.x, y, angles, uv, reference, cfg)
    if after["objective"] > before["objective"] + 1e-8 * max(1, before["objective"]):
        raise RuntimeError("continuation increased objective")
    stationary = after["projected_gradient_inf"] <= design["projected_gradient_tolerance"]
    return dict(
        parameters=result.x.tolist(),
        objective=after["objective"],
        converged=bool(stationary),
        solver_success=bool(result.success),
        status=int(result.status),
        message=result.message,
        nfev=int(result.nfev),
        optimality=float(result.optimality),
        active_indices=np.flatnonzero(result.active_mask).tolist(),
        before=before,
        after=after,
        projection_jacobian_calls=calls,
        elapsed_seconds=time.monotonic() - started,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=resource_path("worm_stationarity_protocol_v1.json"),
    )
    parser.add_argument(
        "--parent", type=Path, default=Path("output/worm_uncertainty/worm_staged_v1")
    )
    parser.add_argument(
        "--pilot", type=Path, default=Path("output/worm_uncertainty/worm_shape_pilot_v1")
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("output/worm_uncertainty/worm_fit_controls_v1/references.npz"),
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
    if sha256(args.parent / "report.json") != design["parent_report_sha256"]:
        raise ValueError("parent hash mismatch")
    parent = json.loads((args.parent / "report.json").read_text())
    for n, h in parent["artifacts_sha256"].items():
        if sha256(args.parent / n) != h:
            raise ValueError("artifact hash mismatch: " + n)
    if sha256(args.pilot / "inputs.npz") != parent["input_sha256"]:
        raise ValueError("input hash mismatch")
    if sha256(args.references) != parent["references_sha256"]:
        raise ValueError("reference hash mismatch")
    data = np.load(args.pilot / "inputs.npz")
    reference = np.load(args.references)["16"]
    cfg = parent["configuration"]
    angles, uv, measured = [data[k] for k in ["angles", "uv", "measured"]]
    mask = measured.reshape(len(angles), len(uv))[:, 0]
    if not np.array_equal(measured, np.repeat(mask, len(uv))):
        raise ValueError("whole acquired views required")
    y = data["noisy"][measured]
    saved = json.loads((args.parent / "fits.json").read_text())
    expected = {(a, s) for a in ["joint_450", "staged_450"] for s in [0, 81, 193, 421]}
    if len(saved) != 8 or {(f["arm"], f["seed"]) for f in saved} != expected:
        raise ValueError("eight frozen candidates required")
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_text(args.protocol.read_text())
    provenance = dict(
        protocol=design,
        parent_report_sha256=sha256(args.parent / "report.json"),
        source_sha256={
            n: sha256(resource_path(n))
            for n in [
                "worm_stationarity.py",
                "staged_density.py",
                "cholesky_gaussian.py",
                "worm_fit_controls.py",
                "prepare.py",
            ]
        },
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    audit = [
        dict(
            arm=f["arm"],
            seed=f["seed"],
            **diagnostics(np.array(f["parameters"]), y, angles[mask], uv, reference, cfg),
        )
        for f in saved
    ]
    (args.output / "initial_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    fits = []
    for f in saved:
        row = continue_fit(np.array(f["parameters"]), y, angles[mask], uv, reference, cfg, design)
        row.update(arm=f["arm"], seed=f["seed"])
        fits.append(row)
        (args.output / "fits.json").write_text(json.dumps(fits, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in [
                        "arm",
                        "seed",
                        "objective",
                        "converged",
                        "status",
                        "nfev",
                        "optimality",
                        "elapsed_seconds",
                    ]
                }
            ),
            flush=True,
        )
    rows = []
    for arm in ["joint_450", "staged_450"]:
        chosen, gate = stability(
            [f for f in fits if f["arm"] == arm], angles[mask], uv, cfg["stability_gate"]
        )
        prediction = projection(np.array(chosen["parameters"]), angles, uv)[0]
        clean = data["clean"]
        errors = {
            k: float(np.linalg.norm((prediction - clean)[m]) / np.linalg.norm(clean[m]))
            for k, m in [("measured_relative_l2", measured), ("withheld_relative_l2", ~measured)]
        }
        rows.append(dict(arm=arm, gate=gate, **errors))
    report = dict(
        **provenance,
        rows=rows,
        artifacts_sha256={
            n: sha256(args.output / n)
            for n in ["fits.json", "initial_audit.json", "protocol.json", "provenance.json"]
        },
        claim="Fixed continuation; convergence means projected-gradient criterion, distinct from the parent solver-termination definition. No posterior claim.",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
