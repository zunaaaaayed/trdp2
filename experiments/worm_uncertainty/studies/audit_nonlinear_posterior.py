"""Bounded nonlinear target audit; fixed small cohort, no withheld-data adaptation."""

from experiments.worm_uncertainty.layout import resource_path

import argparse
import json
import platform
import socket
import time
from pathlib import Path

import numpy as np
from experiments.worm_uncertainty.core.cholesky_gaussian import projection, voxel
from experiments.worm_uncertainty.core.nonlinear_target import (
    Target,
    proposal_logpdf,
    weights,
    weighted_summary,
    split_diagnostics,
    metropolis,
)
from experiments.worm_uncertainty.core.posterior_local import (
    gaussian_factor,
    projection_mean,
    voxel_mean,
)
from experiments.worm_uncertainty.data.prepare import sha256
from experiments.worm_uncertainty.core.staged_density import prior_and_bounds


def load_case(name, parent_dir, comparison_dir):
    parent = json.loads((parent_dir / "report.json").read_text())
    comparison = json.loads((comparison_dir / "report.json").read_text())
    row = next(r for r in comparison["rows"] if r["case"] == name)
    fits = json.loads((parent_dir / "fits.json").read_text())
    chosen = next(f for f in fits if f["case"] == name and f["stage"] == row["selected_stage"])
    alternative = next(
        f
        for f in fits
        if f["case"] == name
        and f["stage"] == ("oracle" if row["selected_stage"] == "final" else "final")
    )
    data = np.load(parent_dir / "inputs.npz")
    truth = data[f"{row['phantom_seed']}_{row['shape']}_truth"]
    mask, sigma = data[name + "_mask"], row["sigma"]
    mean, sd, lower, upper = prior_and_bounds(truth, parent["base_configuration"])
    target = Target(
        data[name + "_y"], data["angles"][mask], data["uv"], sigma, mean, sd, lower, upper
    )
    p = np.array(chosen["parameters"])
    objective = -2 * target.logpdf(p)
    if not np.isclose(objective, chosen["objective"], rtol=1e-12, atol=1e-8):
        raise ValueError("saved objective mismatch")
    queries = np.load(comparison_dir / "queries.npz")
    points, angles, uv = queries["points"], queries["angles"], queries["projection_uv"]
    vt, pt = voxel_mean(truth, points), projection_mean(truth, angles, uv)
    foreground = pt >= 0.05 * pt.max()
    measured = np.repeat(mask, len(uv))
    masks = {
        "density": vt >= 0.05 * vt.max(),
        "acquired": foreground & measured,
        "withheld": foreground & ~measured,
    }
    index = next(i for i, c in enumerate(parent["cases"]) if c["case"] == name)
    return (
        target,
        p,
        np.array(alternative["parameters"]),
        points,
        angles,
        uv,
        vt,
        pt,
        masks,
        index,
        objective,
    )


def summarize(v, pr, vt, pt, masks, w):
    return {
        key: weighted_summary(
            v if key == "density" else pr, vt if key == "density" else pt, mask, w
        )
        for key, mask in masks.items()
    }


def outputs(draws, points, angles, uv):
    return np.array([voxel_mean(p, points) for p in draws]), np.array(
        [projection_mean(p, angles, uv) for p in draws]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=resource_path("nonlinear_posterior_protocol_v1.json"),
    )
    parser.add_argument(
        "--parent", type=Path, default=Path("output/worm_uncertainty/scaled_acceptance_v1")
    )
    parser.add_argument(
        "--comparison", type=Path, default=Path("output/worm_uncertainty/posterior_comparison_v1")
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
        raise ValueError("parent report hash mismatch")
    for folder in [args.parent, args.comparison]:
        report = json.loads((folder / "report.json").read_text())
        for name, h in report["artifacts_sha256"].items():
            if sha256(folder / name) != h:
                raise ValueError("artifact hash mismatch: " + name)
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_text(args.protocol.read_text())
    source_names = [
        "audit_nonlinear_posterior.py",
        "nonlinear_target.py",
        "posterior_local.py",
        "cholesky_gaussian.py",
        "staged_density.py",
        "shape_gaussian.py",
        "rotation_sensitivity.py",
        "stress_3d.py",
        "prepare.py",
    ]
    provenance = dict(
        protocol_sha256=sha256(args.protocol),
        sources={n: sha256(resource_path(n)) for n in source_names},
        parents={
            str(f / "report.json"): sha256(f / "report.json")
            for f in [args.parent, args.comparison]
        },
        host=socket.gethostname(),
        python=platform.python_version(),
        numpy=np.__version__,
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    started = time.monotonic()
    rows = []
    for name in design["cases"]:
        target, p, _, points, angles, uv, vt, pt, masks, index, objective = load_case(
            name, args.parent, args.comparison
        )
        base, jac = projection(p, target.angles, target.uv)
        vb, vj = voxel(p, points)
        pb, pj = projection(p, angles, uv)
        row = dict(case=name, objective=objective, methods={})
        for model, columns in [("density", np.arange(45, 50)), ("joint", np.arange(50))]:
            print(name, model, flush=True)
            factor, _ = gaussian_factor(jac[:, columns], target.sd[columns], target.sigma)
            z = np.random.default_rng(design["seed"] + index).normal(
                size=(design["samples"], len(columns))
            )
            draws = np.tile(p, (len(z), 1))
            draws[:, columns] += z @ factor.T
            logq = proposal_logpdf(draws[:, columns], p[columns], factor)
            valid, ll, lp, lt, discrepancy = [], [], [], [], []
            for draw in draws:
                ok, likelihood, prior, logtarget, pred = target.evaluate(draw)
                valid.append(ok)
                ll.append(likelihood)
                lp.append(prior)
                lt.append(logtarget)
                discrepancy.append(
                    float(np.sqrt(np.mean(((pred - base - jac @ (draw - p)) / target.sigma) ** 2)))
                )
            valid = np.array(valid)
            logw = np.array(lt) - logq
            w, diagnostic = weights(logw)
            v, pr = outputs(draws, points, angles, uv)
            uniform = np.full(len(w), 1 / len(w))
            half = len(w) // 2
            batch_metrics, batch_weights = [], []
            for s in [slice(0, half), slice(half, None)]:
                bw, bd = weights(logw[s])
                batch_weights.append(bd)
                batch_metrics.append(summarize(v[s], pr[s], vt, pt, masks, bw))
            moments = np.column_stack(
                [
                    v[:, masks["density"]].mean(axis=1),
                    pr[:, masks["acquired"]].mean(axis=1),
                    pr[:, masks["withheld"]].mean(axis=1),
                ]
            )
            scale = np.sqrt(w @ (moments - w @ moments) ** 2)
            mean_delta = np.abs(
                weights(logw[:half])[0] @ moments[:half] - weights(logw[half:])[0] @ moments[half:]
            ) / np.maximum(scale, 1e-30)
            width_delta = max(
                abs(batch_metrics[0][k]["width"] - batch_metrics[1][k]["width"])
                / max((batch_metrics[0][k]["width"] + batch_metrics[1][k]["width"]) / 2, 1e-30)
                for k in masks
            )
            rule = design["importance_adequacy"]
            adequate = (
                diagnostic["ess"] >= rule["minimum_ess"]
                and diagnostic["ess_fraction"] >= rule["minimum_ess_fraction"]
                and diagnostic["maximum_weight"] <= rule["maximum_weight"]
                and max(mean_delta) <= rule["maximum_batch_mean_sd_difference"]
                and width_delta <= rule["maximum_batch_width_relative_difference"]
            )
            row["methods"][model] = dict(
                valid_fraction=float(valid.mean()),
                importance=diagnostic,
                adequate=bool(adequate),
                budget_diagnostics={str(n): weights(logw[:n])[1] for n in design["budgets"]},
                batch_weights=batch_weights,
                batch_metrics=batch_metrics,
                batch_mean_sd_difference=mean_delta.tolist(),
                batch_width_relative_difference=width_delta,
                top10_weight=float(np.sort(w)[-10:].sum()),
                nonlinear_discrepancy_sigma_quantiles=np.quantile(
                    discrepancy, [0.5, 0.9, 0.99]
                ).tolist(),
                nonlinear_valid_discrepancy_sigma_quantiles=np.quantile(
                    np.array(discrepancy)[valid], [0.5, 0.9, 0.99]
                ).tolist(),
                local_linear=summarize(
                    vb + (draws - p) @ vj.T, pb + (draws - p) @ pj.T, vt, pt, masks, uniform
                ),
                local_nonlinear=summarize(v, pr, vt, pt, masks, uniform),
                truncated_gaussian=summarize(v, pr, vt, pt, masks, valid.astype(float)),
                weighted=summarize(v, pr, vt, pt, masks, w),
                weighted_status="exploratory adequate screens only"
                if adequate
                else "FAILED: diagnostic only, not reliable corrected intervals",
            )
            np.savez_compressed(
                args.output / (name + "_" + model + ".npz"),
                parameters=draws,
                log_likelihood=ll,
                log_prior=lp,
                log_target=lt,
                log_proposal=logq,
                valid=valid,
                logweights=logw,
                nonlinear_discrepancy_sigma=discrepancy,
                output_moments=moments,
            )
        rows.append(row)
        (args.output / "audit_cases.json").write_text(json.dumps(rows, indent=2))
    result = dict(
        provenance=provenance,
        protocol=design,
        rows=rows,
        elapsed_seconds=time.monotonic() - started,
        reference_needed=any(not r["methods"]["joint"]["adequate"] for r in rows),
    )
    (args.output / "report.json").write_text(json.dumps(result, indent=2))
    print("importance audit complete; reference needed:", result["reference_needed"], flush=True)
    if result["reference_needed"]:
        run_reference(args, design)


def run_reference(args, design):
    started = time.monotonic()
    cfg = design["reference"]
    rows = []
    for name in design["reference_cases"]:
        target, p, alt, points, angles, uv, vt, pt, masks, _, _ = load_case(
            name, args.parent, args.comparison
        )
        _, jac = projection(p, target.angles, target.uv)
        factor, _ = gaussian_factor(jac, target.sd, target.sigma)
        rng = np.random.default_rng(cfg["seed"])
        starts = [p, alt]
        while len(starts) < cfg["chains"]:
            trial = p + 0.25 * factor @ rng.normal(size=len(p))
            if np.isfinite(target.logpdf(trial)):
                starts.append(trial)
        chains, logs, rates = [], [], []
        for i, start in enumerate(starts):
            if time.monotonic() - started > cfg["max_seconds"]:
                rows.append(dict(case=name, status="runtime cap; incomplete, no reference claim"))
                break
            print("reference", name, "chain", i, flush=True)
            draws, logp, rate = metropolis(
                target, p, factor, start, cfg["warmup"] + cfg["draws"], cfg["step"], cfg["seed"] + i
            )
            chains.append(draws[cfg["warmup"] :])
            logs.append(logp[cfg["warmup"] :])
            rates.append(rate)
        if len(chains) != cfg["chains"]:
            break
        chains = np.array(chains)
        logs = np.array(logs)
        flat = chains.reshape(-1, len(p))
        v, pr = outputs(flat, points, angles, uv)
        moments = np.column_stack(
            [
                v[:, masks["density"]].mean(axis=1),
                pr[:, masks["acquired"]].mean(axis=1),
                pr[:, masks["withheld"]].mean(axis=1),
            ]
        )
        white = np.linalg.solve(factor, (flat - p).T).T.reshape(chains.shape)
        diagnostic = split_diagnostics(
            np.concatenate(
                [white, logs[:, :, None], moments.reshape(cfg["chains"], cfg["draws"], 3)], axis=2
            )
        )
        passed = (
            diagnostic["maximum_split_rhat"] <= cfg["maximum_split_rhat"]
            and diagnostic["minimum_ess"] >= cfg["minimum_ess"]
        )
        rows.append(
            dict(
                case=name,
                acceptance_rates=rates,
                diagnostics=diagnostic,
                passed=bool(passed),
                status="feasibility screens passed; rank/tail diagnostics and longer replication still required"
                if passed
                else "FAILED mixing screens: not a validated reference posterior",
                metrics=summarize(v, pr, vt, pt, masks, np.ones(len(flat))),
                chain_metrics=[
                    summarize(
                        v[i * cfg["draws"] : (i + 1) * cfg["draws"]],
                        pr[i * cfg["draws"] : (i + 1) * cfg["draws"]],
                        vt,
                        pt,
                        masks,
                        np.ones(cfg["draws"]),
                    )
                    for i in range(cfg["chains"])
                ],
                bounds_violations=int(
                    ((flat < target.lower) | (flat > target.upper)).any(axis=1).sum()
                ),
            )
        )
        np.savez_compressed(
            args.output / (name + "_reference.npz"),
            parameters=chains,
            log_target=logs,
            output_moments=moments,
            starts=starts,
        )
        (args.output / "reference.json").write_text(
            json.dumps(dict(rows=rows, elapsed_seconds=time.monotonic() - started), indent=2)
        )
    (args.output / "reference.json").write_text(
        json.dumps(dict(rows=rows, elapsed_seconds=time.monotonic() - started), indent=2)
    )


if __name__ == "__main__":
    main()
