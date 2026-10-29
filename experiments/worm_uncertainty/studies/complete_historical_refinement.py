"""Append the one missing v2 fit without altering historical reports or thresholds."""

import argparse
import json
import signal
import time
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.core.cholesky_gaussian import fit_control, projection
from experiments.worm_uncertainty.core.measured_initializer import perturb
from experiments.worm_uncertainty.core.scaled_acceptance import assess
from experiments.worm_uncertainty.core.staged_density import prior_and_bounds
from experiments.worm_uncertainty.studies.worm_fit_controls import stability
from experiments.worm_uncertainty.data.prepare import sha256


def write(p, obj):
    p.write_text(json.dumps(obj, indent=2) + "\n")


def acceptance_parts(row, reference, cfg, thresholds):
    p = np.asarray(row["parameters"])
    _, _, lo, hi = prior_and_bounds(reference, cfg)
    n = row["numerical"]
    return dict(
        finite=bool(np.isfinite(p).all() and np.isfinite(row["objective"])),
        bounds=bool(np.all(p >= lo) and np.all(p <= hi)),
        optimizer_terminated=row["converged"],
        stationarity={
            key: dict(value=n[key], threshold=thresholds[key], passed=n[key] <= thresholds[key])
            for key in [
                "normalized_gradient",
                "relative_predicted_decrease",
                "prior_scaled_correction_inf",
            ]
        },
        numerical_accepted=n["accepted"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("output/worm_uncertainty/completion_refinement_v2_retry1"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    original = json.loads((args.parent / "report.json").read_text())
    for name, digest in original["artifacts_sha256"].items():
        if sha256(args.parent / name) != digest:
            raise ValueError("historical artifact changed")
    protocol = json.loads((args.parent / "protocol.json").read_text())
    cfg = protocol["configuration"]
    fits = json.loads((args.parent / "fits.json").read_text())
    expected = {
        (arm, seed)
        for arm in ["acquired", "interpolation", "learned"]
        for seed in protocol["integration"]["seeds"]
    }
    missing = expected - {(r["arm"], r["seed"]) for r in fits}
    if missing != {("learned", 421)}:
        raise ValueError("unexpected missing fit identities")
    a = np.load(args.parent / "inputs.npz")
    args.output.mkdir(parents=True)
    write(
        args.output / "protocol.json",
        dict(
            parent_report_sha256=sha256(args.parent / "report.json"),
            missing=["learned", 421],
            max_wall_seconds=180,
            cfg=cfg,
            thresholds=protocol["thresholds"],
            source_sha256=sha256(Path(__file__)),
        ),
    )
    started = time.monotonic()

    def timeout(*_):
        raise TimeoutError("single historical fit exceeded 180 seconds")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(180)
    try:
        fit, _ = fit_control(
            a["learned_observations"],
            a["angles"][a["mask"]],
            a["uv"],
            perturb(a["learned_initial"], 421, cfg),
            a["reference"],
            cfg["noise_sigma"],
            cfg,
            False,
        )
        n = assess(
            fit.x,
            a["learned_observations"],
            a["angles"][a["mask"]],
            a["uv"],
            a["reference"],
            cfg,
            protocol["thresholds"],
        )
        new = dict(
            arm="learned",
            seed=421,
            parameters=fit.x.tolist(),
            objective=float(2 * fit.cost),
            converged=bool(fit.success),
            status=int(fit.status),
            nfev=int(fit.nfev),
            active_indices=np.flatnonzero(fit.active_mask).tolist(),
            numerical=n,
            elapsed_seconds=time.monotonic() - started,
        )
        fits.append(new)
        write(args.output / "appended_fit.json", new)
    except TimeoutError as exc:
        write(
            args.output / "failure.json",
            dict(reason=str(exc), elapsed_seconds=time.monotonic() - started),
        )
        return
    finally:
        signal.alarm(0)
    rows = []
    predictions = {}
    for arm in ["acquired", "interpolation", "learned"]:
        candidates = [r for r in fits if r["arm"] == arm]
        chosen, gate = stability(candidates, a["angles"][a["mask"]], a["uv"], cfg["stability_gate"])
        initial = projection(a[arm + "_initial"], a["angles"], a["uv"])[0].reshape(
            a["target"].shape
        )
        pred = projection(np.array(chosen["parameters"]), a["angles"], a["uv"])[0].reshape(
            a["target"].shape
        )
        predictions[arm] = pred
        errors = {
            stage: {
                key: float(
                    np.linalg.norm((value - a["target"])[mask]) / np.linalg.norm(a["target"][mask])
                )
                for key, mask in [("acquired", a["mask"]), ("withheld", ~a["mask"])]
            }
            for stage, value in [("initialization", initial), ("refined", pred)]
        }
        rows.append(
            dict(
                arm=arm,
                selected_seed=chosen["seed"],
                objective=chosen["objective"],
                starts=[r["seed"] for r in candidates],
                gate=gate,
                acceptance=acceptance_parts(chosen, a["reference"], cfg, protocol["thresholds"]),
                errors=errors,
            )
        )
    np.savez_compressed(args.output / "predictions.npz", **predictions)
    write(
        args.output / "report.json",
        dict(
            rows=rows,
            elapsed_seconds=time.monotonic() - started,
            complete_start_intersection=sorted(
                set.intersection(
                    *[
                        {r["seed"] for r in fits if r["arm"] == arm}
                        for arm in ["acquired", "interpolation", "learned"]
                    ]
                )
            ),
            original_report_preserved=True,
            artifacts_sha256={p.name: sha256(p) for p in args.output.iterdir()},
        ),
    )
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
