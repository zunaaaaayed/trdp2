"""Frozen local posterior propagation comparison on numerically eligible phantoms."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc
from cholesky_gaussian import projection, voxel
from gaussian_l2 import relative_l2
from posterior_local import gaussian_factor, interval_metrics, projection_mean, voxel_mean
from prepare import sha256
from scaled_acceptance import assess
from staged_density import prior_and_bounds


def summarize_samples(samples, truth, masks):
    result = {}
    for name, mask in masks.items():
        metrics, _, _, _ = interval_metrics(samples, truth, mask)
        mean = np.mean(samples, axis=0)
        metrics["relative_l2"] = float(
            np.linalg.norm((mean - truth)[mask]) / max(np.linalg.norm(truth[mask]), 1e-30)
        )
        metrics["width_relative_peak"] = metrics["mean_width95"] / max(float(np.max(truth)), 1e-30)
        halves = [interval_metrics(part, truth, mask)[0] for part in np.array_split(samples, 2)]
        metrics["half_sample_coverage_difference"] = abs(
            halves[0]["coverage95"] - halves[1]["coverage95"]
        )
        metrics["half_sample_width_relative_difference"] = abs(
            halves[0]["mean_width95"] - halves[1]["mean_width95"]
        ) / max(metrics["mean_width95"], 1e-30)
        result[name] = metrics
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("posterior_comparison_protocol_v1.json"),
    )
    parser.add_argument(
        "--parent", type=Path, default=Path("output/worm_uncertainty/scaled_acceptance_v1")
    )
    parser.add_argument(
        "--eligibility",
        type=Path,
        default=Path("docs/figures/worm_uncertainty/scaled_acceptance_v1/summary.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    design = json.loads(args.protocol.read_text())
    if not args.execute:
        print(json.dumps(design, indent=2))
        return
    if args.output.exists():
        raise ValueError("output exists")
    if sha256(args.parent / "report.json") != design["parent_report_sha256"]:
        raise ValueError("parent hash mismatch")
    if sha256(args.eligibility) != design["eligibility_summary_sha256"]:
        raise ValueError("eligibility hash mismatch")
    parent = json.loads((args.parent / "report.json").read_text())
    eligible = json.loads(args.eligibility.read_text())
    for n, h in parent["artifacts_sha256"].items():
        if sha256(args.parent / n) != h:
            raise ValueError("artifact hash mismatch: " + n)
    fits = json.loads((args.parent / "fits.json").read_text())
    data = np.load(args.parent / "inputs.npz")
    points = qmc.Sobol(3, scramble=True, seed=design["query_seed"]).random_base2(
        design["query_sobol_power"]
    )
    points = points * np.array([2.0, 2.0, 0.32]) + np.array([-1.0, -1.0, -0.16])
    axis = np.linspace(-1, 1, 65)
    xy = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    slice_points = np.column_stack([xy, np.zeros(len(xy))])
    angles = data["angles"]
    uv = data["uv"]
    small_uv = uv.reshape(9, 129, 2)[::2, ::4].reshape(-1, 2)
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_text(args.protocol.read_text())
    np.savez_compressed(
        args.output / "queries.npz",
        points=points,
        angles=angles,
        projection_uv=small_uv,
        slice_axis=axis,
    )
    sources = [
        "posterior_comparison.py",
        "posterior_local.py",
        "cholesky_gaussian.py",
        "gaussian_l2.py",
        "scaled_acceptance.py",
        "staged_density.py",
        "prepare.py",
        "shape_gaussian.py",
        "rotation_sensitivity.py",
        "stress_3d.py",
    ]
    provenance = dict(
        protocol=design,
        source_sha256={n: sha256(Path(__file__).with_name(n)) for n in sources},
        selection="Eligible final/oracle minimum acquired objective; ground truth used only in evaluation.",
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    rows = []
    excluded = []
    covariances = {}
    visual = {}
    started = time.monotonic()
    for index, case in enumerate(parent["cases"]):
        name = case["case"]
        pair = [
            next(f for f in fits if f["case"] == name and f["stage"] == s)
            for s in ["final", "oracle"]
        ]
        mask = data[name + "_mask"]
        y = data[name + "_y"]
        truth = data[f"{case['phantom_seed']}_{case['shape']}_truth"]
        clean_full = data[f"{case['phantom_seed']}_{case['shape']}_clean"]
        sigma = float(
            parent["protocol"]["relative_noise_sd"] * np.sqrt(np.mean(clean_full[mask] ** 2))
        )
        cfg = dict(parent["base_configuration"], noise_sigma=sigma, components=5)
        checks = [
            assess(
                np.array(f["parameters"]),
                y,
                angles[mask],
                uv,
                truth,
                cfg,
                parent["protocol"]["thresholds"],
            )
            for f in pair
        ]
        predictions = [projection_mean(np.array(f["parameters"]), angles[mask], uv) for f in pair]
        gap = abs(pair[0]["objective"] - pair[1]["objective"]) / max(
            1.0, min(f["objective"] for f in pair)
        )
        disagreement = float(
            np.linalg.norm(predictions[0] - predictions[1])
            / max(np.linalg.norm(predictions[0]), 1e-30)
        )
        gate = cfg["stability_gate"]
        passed = (
            all(c["accepted"] and not c["active_indices"] for c in checks)
            and gap <= gate["relative_objective_gap"]
            and disagreement <= gate["relative_prediction_disagreement"]
        )
        recorded = next(r["passed"] for r in eligible["multistart"] if r["case"] == name)
        if passed != recorded:
            raise ValueError("eligibility recomputation mismatch")
        if not passed:
            excluded.append(
                dict(
                    case=name,
                    reason="failed independent-start agreement",
                    objective_gap=gap,
                    prediction_disagreement=disagreement,
                )
            )
            continue
        chosen = min(pair, key=lambda f: f["objective"])
        p = np.array(chosen["parameters"])
        k = len(p) // 10
        _, sd, lower, upper = prior_and_bounds(truth, cfg)
        _, jac = projection(p, angles[mask], uv)
        factors = {}
        conditions = {}
        for model, columns in [("density", np.arange(9 * k, 10 * k)), ("joint", np.arange(10 * k))]:
            f, condition = gaussian_factor(jac[:, columns], sd[columns], sigma)
            expanded = np.zeros((len(p), len(columns)))
            expanded[columns] = f
            factors[model] = expanded
            conditions[model] = condition
            covariances[name + "_" + model] = expanded @ expanded.T
        marginal_difference = (
            covariances[name + "_joint"][9 * k :, 9 * k :]
            - covariances[name + "_density"][9 * k :, 9 * k :]
        )
        if np.linalg.eigvalsh(marginal_difference).min() < -1e-8:
            raise ValueError("joint density variance unexpectedly smaller")
        truth_voxel = voxel_mean(truth, points)
        truth_projection = projection_mean(truth, angles, small_uv)
        vmasks = dict(
            foreground=truth_voxel >= design["foreground_fraction"] * truth_voxel.max(),
            all=np.ones(len(points), dtype=bool),
        )
        pmask = np.repeat(mask, len(small_uv))
        foreground = truth_projection >= design["foreground_fraction"] * truth_projection.max()
        pmasks = dict(
            acquired_foreground=pmask & foreground, withheld_foreground=(~pmask) & foreground
        )
        baseline_v, jv = voxel(p, points)
        baseline_p, jp = projection(p, angles, small_uv)
        is_visual = name == design["designated_visual_case"]
        if is_visual:
            visual["truth"] = voxel_mean(truth, slice_points)
            visual["map"] = voxel_mean(p, slice_points)
            slice_base, slice_jac = voxel(p, slice_points)
        row = dict(
            case=name,
            shape=case["shape"],
            phantom_seed=case["phantom_seed"],
            offset=case["offset"],
            noise_seed=case["noise_seed"],
            selected_stage=chosen["stage"],
            sigma=sigma,
            map_exact_density_error=relative_l2(p, truth),
            methods={},
            bounds={},
            precision_conditions=conditions,
            voxel_query_counts={key: int(value.sum()) for key, value in vmasks.items()},
            projection_query_counts={key: int(value.sum()) for key, value in pmasks.items()},
        )
        for model in ["density", "joint"]:
            factor = factors[model]
            z = np.random.default_rng(design["sample_seed"] + index).normal(
                size=(design["samples"], factor.shape[1])
            )
            draws = p[None, :] + z @ factor.T
            violations = (draws < lower[None, :]) | (draws > upper[None, :])
            row["bounds"][model] = dict(
                any_fraction=float(np.mean(violations.any(axis=1))),
                per_parameter_fraction=np.mean(violations, axis=0).tolist(),
                negative_density_fraction=float(np.mean((draws[:, 9 * k :] < 0).any(axis=1))),
                approximation_flag=bool(np.mean(violations.any(axis=1)) > 0.01),
            )
            deltas = draws - p[None, :]
            linear_v = baseline_v[None, :] + deltas @ jv.T
            linear_p = baseline_p[None, :] + deltas @ jp.T
            nonlinear_v = np.empty_like(linear_v)
            nonlinear_p = np.empty_like(linear_p)
            nonlinear_slice = np.empty((len(draws), len(slice_points))) if is_visual else None
            for n, draw in enumerate(draws):
                nonlinear_v[n] = voxel_mean(draw, points)
                nonlinear_p[n] = projection_mean(draw, angles, small_uv)
                if is_visual:
                    nonlinear_slice[n] = voxel_mean(draw, slice_points)
            if model == "density":
                if not np.allclose(
                    linear_v, nonlinear_v, rtol=1e-10, atol=1e-12
                ) or not np.allclose(linear_p, nonlinear_p, rtol=1e-10, atol=1e-12):
                    raise ValueError("density propagation should be exactly linear")
            for propagation, vs, ps in [
                ("linear", linear_v, linear_p),
                ("nonlinear", nonlinear_v, nonlinear_p),
            ]:
                key = model + "_" + propagation
                row["methods"][key] = dict(
                    voxel=summarize_samples(vs, truth_voxel, vmasks),
                    projection=summarize_samples(ps, truth_projection, pmasks),
                )
                if is_visual:
                    ss = (
                        slice_base[None, :] + deltas @ slice_jac.T
                        if propagation == "linear"
                        else nonlinear_slice
                    )
                    visual[key + "_mean"] = np.mean(ss, axis=0)
                    visual[key + "_sd"] = np.std(ss, axis=0, ddof=1)
                    visual[key + "_lo"], visual[key + "_hi"] = np.quantile(
                        ss, [0.025, 0.975], axis=0
                    )
            row.setdefault("linear_monte_carlo_sd_relative_error", {})[model] = float(
                np.median(
                    abs(np.std(linear_v, axis=0, ddof=1) - np.linalg.norm(jv @ factor, axis=1))
                    / np.maximum(np.linalg.norm(jv @ factor, axis=1), 1e-30)
                )
            )
        rows.append(row)
        (args.output / "cases.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    completed=len(rows),
                    case=name,
                    coverage={
                        m: row["methods"][m]["voxel"]["foreground"]["coverage95"]
                        for m in row["methods"]
                    },
                    bounds=row["bounds"]["joint"]["any_fraction"],
                )
            ),
            flush=True,
        )
    if len(rows) != 31 or len(excluded) != 1:
        raise ValueError("frozen cohort accounting mismatch")
    np.savez_compressed(args.output / "covariances.npz", **covariances)
    np.savez_compressed(args.output / "visual.npz", **visual)
    report = dict(
        **provenance,
        rows=rows,
        excluded=excluded,
        eligible_clean=sum(r["noise_seed"] is None for r in rows),
        eligible_noisy=sum(r["noise_seed"] is not None for r in rows),
        elapsed_seconds=time.monotonic() - started,
        artifacts_sha256={
            n: sha256(args.output / n)
            for n in [
                "cases.json",
                "queries.npz",
                "covariances.npz",
                "visual.npz",
                "protocol.json",
                "provenance.json",
            ]
        },
        claim="Conditional on numerical eligibility and an oracle geometry prior; local untruncated Gauss-Newton approximation. Coverage is empirical over correlated evaluation points, not a certification.",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
