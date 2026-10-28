"""One bounded MALA feasibility run; separate tuning and frozen evaluation settings."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.studies.audit_nonlinear_posterior import load_case, outputs
from experiments.worm_uncertainty.core.cholesky_gaussian import projection
from experiments.worm_uncertainty.core.posterior_local import gaussian_factor
from experiments.worm_uncertainty.core.langevin_reference import (
    AffineTarget,
    mala,
    rank_diagnostics,
)
from experiments.worm_uncertainty.data.prepare import sha256


def write(p, x):
    p.write_text(json.dumps(x, indent=2) + "\n")


def setup(name):
    loaded = load_case(
        name,
        Path("output/worm_uncertainty/scaled_acceptance_v1"),
        Path("output/worm_uncertainty/posterior_comparison_v1"),
    )
    target, p = loaded[:2]
    _, j = projection(p, target.angles, target.uv)
    f, _ = gaussian_factor(j, target.sd, target.sigma)
    return loaded, AffineTarget(target, p, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "protocols/langevin_feasibility_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.protocol.read_text())
    if not args.execute:
        print(json.dumps(cfg, indent=2))
        return
    if args.output.exists():
        raise ValueError("output exists")
    started = time.monotonic()
    deadline = started + cfg["max_seconds"]
    args.output.mkdir(parents=True)
    write(args.output / "protocol.json", cfg)
    write(
        args.output / "provenance.json",
        dict(
            protocol_sha256=sha256(args.protocol),
            sources={str(p): sha256(p) for p in Path("experiments/worm_uncertainty").rglob("*.py")},
            parent_reports={
                str(p): sha256(p)
                for p in [
                    Path("output/worm_uncertainty/scaled_acceptance_v1/report.json"),
                    Path("output/worm_uncertainty/posterior_comparison_v1/report.json"),
                ]
            },
        ),
    )
    _, t = setup(cfg["tuning_case"])
    tuning = []
    for step in cfg["step_candidates"]:
        x, lp, diagnostic = mala(
            t.value_gradient, np.zeros(50), step, cfg["tuning_steps"], cfg["seed"], deadline
        )
        tuning.append(dict(step=step, **diagnostic))
        np.savez_compressed(args.output / f"tuning_{step}.npz", q=x, log_target=lp)
    eligible = [
        r
        for r in tuning
        if 0.3 <= r["acceptance"] <= 0.9 and r["completed_steps"] == cfg["tuning_steps"]
    ]
    step = (
        max(eligible, key=lambda r: r["mean_squared_jump"])["step"]
        if eligible
        else min(cfg["step_candidates"])
    )
    write(
        args.output / "frozen_settings.json",
        dict(
            tuning_case=cfg["tuning_case"],
            tuning=tuning,
            selected_step=step,
            selection="accepted squared jump, acquired target only",
        ),
    )
    frozen = sha256(args.output / "frozen_settings.json")
    rows = []
    for name in cfg["evaluation_cases"]:
        if time.monotonic() >= deadline:
            break
        loaded, t = setup(name)
        target, p, alt, points, angles, uv, vt, pt, masks, _, _ = loaded
        starts = [np.zeros(50), np.linalg.solve(t.factor, alt - p)]
        rng = np.random.default_rng(cfg["seed"])
        while len(starts) < cfg["chains"]:
            q = rng.normal(0, 0.1, 50)
            if np.isfinite(t.value_gradient(q)[0]):
                starts.append(q)
        chains = []
        logs = []
        records = []
        case_started = time.monotonic()
        for i, q in enumerate(starts):
            x, lp, d = mala(
                t.value_gradient,
                q,
                step,
                cfg["warmup"] + cfg["draws"],
                cfg["seed"] + i + 1,
                deadline,
            )
            np.savez_compressed(
                args.output / (name + f"_chain{i}.npz"), q=x, log_target=lp, start=q
            )
            records.append(d)
            print(name, "chain", i, d, flush=True)
            if len(x) != cfg["warmup"] + cfg["draws"]:
                break
            chains.append(x[cfg["warmup"] :])
            logs.append(lp[cfg["warmup"] :])
        if len(chains) != cfg["chains"]:
            rows.append(
                dict(
                    case=name,
                    status="deadline; incomplete; no reference claim",
                    chains=records,
                    passed=False,
                )
            )
            break
        q = np.array(chains)
        flat = p + q.reshape(-1, 50) @ t.factor.T
        v, pr = outputs(flat, points, angles, uv)
        moments = np.column_stack(
            [
                v[:, masks["density"]].mean(axis=1),
                pr[:, masks["acquired"]].mean(axis=1),
                pr[:, masks["withheld"]].mean(axis=1),
            ]
        )
        values = np.concatenate(
            [q, np.array(logs)[:, :, None], moments.reshape(cfg["chains"], cfg["draws"], 3)], axis=2
        )
        diagnostics = rank_diagnostics(values)
        valid = bool(np.all(flat >= target.lower) and np.all(flat <= target.upper))
        np.savez_compressed(
            args.output / (name + "_outputs.npz"), moments=moments, diagnostic_values=values
        )
        rows.append(
            dict(
                case=name,
                step=step,
                chains=records,
                diagnostics=diagnostics,
                all_bounds_valid=valid,
                passed=bool(valid and diagnostics["passed"]),
                elapsed_seconds=time.monotonic() - case_started,
                status="diagnostics pass; conditional local reference candidate only"
                if diagnostics["passed"]
                else "FAILED convergence: no posterior interval claim",
            )
        )
        write(args.output / "cases.json", rows)
    if frozen != sha256(args.output / "frozen_settings.json"):
        raise ValueError("evaluation altered settings")
    report = dict(
        tuning=tuning,
        selected_step=step,
        rows=rows,
        elapsed_seconds=time.monotonic() - started,
        completed=len(rows) == len(cfg["evaluation_cases"]),
        scientific_claim="one bounded feasibility; failure pauses sampler expansion",
        artifacts_sha256={p.name: sha256(p) for p in args.output.iterdir() if p.is_file()},
    )
    write(args.output / "report.json", report)


if __name__ == "__main__":
    main()
