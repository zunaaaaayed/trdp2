"""Paired worm initialization, prior, solver-budget and capacity diagnostics.

Uses the preserved pilot inputs. No clean or withheld values enter initialization,
optimization, or start selection. No uncertainty intervals are certified here.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from cholesky_gaussian import fit_control, projection
from diagnose_3d import ray_basis
from measured_initializer import initialize, perturb
from prepare import sha256


def multiscale_initialize(y, angles, uv, v, cfg, design):
    """Build a fixed full-field dictionary; chunk derivative temporaries."""
    xy = np.linspace(-cfg["initialization_extent"], cfg["initialization_extent"], design["grid_xy"])
    z = np.linspace(v.min(), v.max(), cfg["initialization_grid_z"])
    grid = np.stack(np.meshgrid(xy, xy, z, indexing="ij"), axis=-1).reshape(-1, 3)
    centres = np.tile(grid, (len(design["widths_xy"]), 1))
    scales = np.concatenate(
        [
            np.tile([width, width, cfg["initialization_width_z"]], (len(grid), 1))
            for width in design["widths_xy"]
        ]
    )
    basis = np.empty((len(y), len(centres)))
    for start in range(0, len(centres), design["dictionary_chunk"]):
        end = start + design["dictionary_chunk"]
        basis[:, start:end] = ray_basis(centres[start:end], scales[start:end], angles, uv)[0]
    local = dict(cfg, minimum_centre_distance=design["minimum_centre_distance"])
    reference, info = initialize(
        y, angles, uv, local, (centres, basis, np.linalg.norm(basis, axis=0))
    )
    k = cfg["components"]
    reference[3 * k : 6 * k] = np.log(scales[info["selected_indices"]]).ravel()
    prediction = projection(reference, angles, uv)[0]
    info.update(
        dictionary_atoms=len(centres),
        dictionary_bytes=basis.nbytes,
        active_densities=int(np.count_nonzero(reference[9 * k :] > 0.01000001)),
        realized_relative_residual=float(np.linalg.norm(prediction - y) / np.linalg.norm(y)),
    )
    return reference, info


def align_to_reference(initial, reference):
    """Match labels by centre distance, without changing represented projections."""
    k = len(reference) // 10
    if initial.shape != reference.shape:
        raise ValueError("matching component counts required")
    a, b = reference[: 3 * k].reshape(k, 3), initial[: 3 * k].reshape(k, 3)
    _, order = linear_sum_assignment(np.sum((a[:, None] - b[None, :]) ** 2, axis=-1))
    return np.concatenate(
        [initial[start * k : (start + 3) * k].reshape(k, 3)[order].ravel() for start in [0, 3, 6]]
        + [initial[9 * k :][order]]
    )


def stability(fits, angles, uv, thresholds):
    converged = sorted([r for r in fits if r["converged"]], key=lambda r: r["objective"])
    chosen = converged[0] if converged else min(fits, key=lambda r: r["objective"])
    gate = dict(passed=False, converged_candidates=len(converged), selected_seed=chosen["seed"])
    if len(converged) >= thresholds["minimum_converged"]:
        first, second = converged[:2]
        a = projection(np.array(first["parameters"]), angles, uv)[0]
        b = projection(np.array(second["parameters"]), angles, uv)[0]
        gap = abs(first["objective"] - second["objective"]) / max(1, abs(first["objective"]))
        disagreement = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-15))
        gate.update(
            relative_objective_gap=gap,
            relative_prediction_disagreement=disagreement,
            passed=not (first["active_indices"] or second["active_indices"])
            and gap <= thresholds["relative_objective_gap"]
            and disagreement <= thresholds["relative_prediction_disagreement"],
        )
    return chosen, gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent", type=Path, default=Path("output/worm_uncertainty/worm_shape_pilot_v1")
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("worm_fit_controls_protocol_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    design = json.loads(args.protocol.read_text())
    if not args.execute:
        print(json.dumps(design, indent=2))
        return
    if args.output.exists():
        raise ValueError("output exists; preserve it and use a fresh path")
    if sha256(args.parent / "report.json") != design["parent_report_sha256"]:
        raise ValueError("parent report hash mismatch")
    parent = json.loads((args.parent / "report.json").read_text())
    for name in ["inputs.npz", "fits.json"]:
        if sha256(args.parent / name) != parent["artifacts_sha256"][name]:
            raise ValueError("parent artifact hash mismatch: " + name)
    data = np.load(args.parent / "inputs.npz")
    angles, uv, v, measured = [data[key] for key in ["angles", "uv", "v", "measured"]]
    angle_mask = measured.reshape(len(angles), len(uv))[:, 0]
    if not np.array_equal(measured, np.repeat(angle_mask, len(uv))):
        raise ValueError("complete acquired views required")
    acquired = data["noisy"][measured]
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_text(args.protocol.read_text())
    references, initializers, fits, rows, predictions = {}, {}, [], [], {}
    started = time.monotonic()
    for k in sorted({arm["components"] for arm in design["arms"]}):
        cfg = dict(parent["protocol"], components=k)
        references[k], initializers[str(k)] = multiscale_initialize(
            acquired, angles[angle_mask], uv, v, cfg, design
        )
        print(json.dumps(dict(components=k, initializer=initializers[str(k)])), flush=True)
    np.savez_compressed(
        args.output / "references.npz", **{str(k): p for k, p in references.items()}
    )
    for arm in design["arms"]:
        cfg = dict(parent["protocol"], components=arm["components"], budget_nfev=arm["budget_nfev"])
        reference = references[arm["components"]]
        initial = reference.copy()
        if arm["prior"] == "parent":
            reference = data["reference"]
            initial = align_to_reference(initial, reference)
        arm_fits = []
        for seed in cfg["initialization_seeds"]:
            fit, _ = fit_control(
                acquired,
                angles[angle_mask],
                uv,
                perturb(initial, seed, cfg),
                reference,
                cfg["noise_sigma"],
                cfg,
                False,
            )
            row = dict(
                arm=arm["name"],
                seed=seed,
                objective=float(2 * fit.cost),
                converged=bool(fit.success),
                nfev=fit.nfev,
                optimality=float(fit.optimality),
                active_indices=np.flatnonzero(fit.active_mask).tolist(),
                parameters=fit.x.tolist(),
            )
            fits.append(row)
            arm_fits.append(row)
            (args.output / "fits.json").write_text(json.dumps(fits, indent=2) + "\n")
            print(json.dumps({k: val for k, val in row.items() if k != "parameters"}), flush=True)
        chosen, gate = stability(arm_fits, angles[angle_mask], uv, cfg["stability_gate"])
        mean = projection(np.array(chosen["parameters"]), angles, uv)[0]
        predictions[arm["name"]] = mean
        clean = data["clean"]
        row = dict(
            arm=arm,
            gate=gate,
            selected_objective=chosen["objective"],
            measured_relative_l2=float(
                np.linalg.norm((mean - clean)[measured]) / np.linalg.norm(clean[measured])
            ),
            withheld_relative_l2=float(
                np.linalg.norm((mean - clean)[~measured]) / np.linalg.norm(clean[~measured])
            ),
        )
        rows.append(row)
    np.savez_compressed(args.output / "predictions.npz", **predictions)
    report = dict(
        protocol=design,
        initializers=initializers,
        rows=rows,
        elapsed_seconds=time.monotonic() - started,
        parent_report_sha256=sha256(args.parent / "report.json"),
        code_sha256=sha256(__file__),
        protocol_sha256=sha256(args.protocol),
        dependencies_sha256={
            name: sha256(Path(__file__).with_name(name))
            for name in [
                "cholesky_gaussian.py",
                "shape_gaussian.py",
                "rotation_sensitivity.py",
                "stress_3d.py",
                "measured_initializer.py",
                "diagnose_3d.py",
                "prepare.py",
            ]
        },
        artifacts_sha256={
            name: sha256(args.output / name)
            for name in ["fits.json", "references.npz", "predictions.npz"]
        },
        claim="One development worm/noise/window; failed gates prohibit calibrated-posterior claims. Capacity changes also change prior dimension.",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
