"""Conditional balanced selection-only Gaussian refinement with equal-start accounting."""

import argparse
import json
import signal
import time
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.data.acquisition_diversity import read_design
from experiments.worm_uncertainty.data.matched_completion import grid
from experiments.worm_uncertainty.core.completion_initialization import initialize_arms
from experiments.worm_uncertainty.core.completion_evaluation import common_starts, select_fit
from experiments.worm_uncertainty.core.cholesky_gaussian import fit_control, projection
from experiments.worm_uncertainty.core.measured_initializer import perturb
from experiments.worm_uncertainty.core.scaled_acceptance import assess
from experiments.worm_uncertainty.studies.complete_historical_refinement import acceptance_parts
from experiments.worm_uncertainty.studies.acquisition_diversity import write, provenance
from experiments.worm_uncertainty.data.prepare import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("output/worm_uncertainty/acquisition_diversity_v3")
    )
    args = parser.parse_args()
    d, _ = read_design()
    parent = args.root / "primary"
    selection = json.loads((parent / "selection.json").read_text())
    selected = selection["selected_arm"]
    folder = args.root / "refinement"
    folder.mkdir(exist_ok=False)
    if selected is None:
        write(
            folder / "report.json",
            dict(skipped=True, reason="no learned arm passed frozen completion gate"),
        )
        return
    frozen = json.loads((args.root / "analysis_freeze.json").read_text())
    if frozen["selection_sha256"] != sha256(parent / "selection.json"):
        raise ValueError("selection changed")
    historical = json.loads((args.root / "historical_completion/report.json").read_text())
    remaining = 1800 - historical["elapsed_seconds"]
    estimate = 12 * historical["elapsed_seconds"] + 30
    if estimate > remaining:
        raise ValueError("preflight insufficient CPU budget for balanced2-case design")
    begun = time.monotonic()
    deadline = begun + remaining
    provenance(folder, d)
    base = Path(__file__).resolve().parents[1] / "protocols"
    cfg = json.loads((base / "worm_shape_protocol_v1.json").read_text())
    cfg.update(components=16, budget_nfev=450, noise_sigma=d["noise_sd"])
    dictionary = json.loads((base / "worm_fit_controls_protocol_v1.json").read_text())
    thresholds = json.loads((base / "scaled_acceptance_protocol_v1.json").read_text())["thresholds"]
    clean = np.load(parent / "selection_clean.npy")
    inputs = np.load(parent / "selection_inputs.npz")
    ensemble = np.mean(
        [
            np.load(parent / f"{selected}_{seed}" / "selection_prediction.npy")
            for seed in d["training_seeds"]
        ],
        axis=0,
    )
    degrees, uv, v = grid(d)
    angles = np.deg2rad(degrees)
    mi = next(
        i
        for i, r in enumerate(d["inventories"]["selection"])
        if r["group"] == "new_in_distribution" and r["count"] == 13
    )
    row = d["inventories"]["selection"][mi]
    mask = np.array(row["mask"], bool)
    names = [d["roles"]["selection"][0], d["roles"]["selection"][-1]]
    write(
        folder / "integration_protocol.json",
        dict(
            cases=names,
            mask=row,
            selected_arm=selected,
            selection_sha256=sha256(parent / "selection.json"),
            configuration=cfg,
            thresholds=thresholds,
            dictionary=dictionary,
            seeds=[0, 81],
            per_fit_seconds=100,
            remaining_seconds=remaining,
            preflight_estimated_seconds=estimate,
            prior="shared acquired-only initializer",
            ensemble="3-seed mean with exact measured reinsertion",
            selection_rule=d["gaussian"]["selection"],
        ),
    )
    reports = []

    def timeout(*_):
        raise TimeoutError("per-fit100second cap")

    signal.signal(signal.SIGALRM, timeout)
    for name in names:
        fi = d["roles"]["selection"].index(name)
        case = folder / name.removesuffix(".tiff")
        case.mkdir()
        observed = inputs["observed"][fi, mi].transpose(1, 0, 2)
        truth = clean[fi].transpose(1, 0, 2)
        interp = inputs["interpolation"][fi, mi].transpose(1, 0, 2)
        learned = ensemble[fi, mi].transpose(1, 0, 2)
        learned[mask] = observed[mask]
        initial, info, contracts = initialize_arms(
            observed, interp, learned, mask, angles, uv, v, cfg, dictionary
        )
        np.savez_compressed(
            case / "inputs.npz",
            observed=observed,
            target=truth,
            interpolation=interp,
            learned=learned,
            mask=mask,
            angles=angles,
            uv=uv,
            v=v,
            reference=contracts["acquired"]["reference"],
            **{a + "_initial": p for a, p in initial.items()},
            **{a + "_observations": c["y"] for a, c in contracts.items()},
        )
        fits = []
        # Seed-major order keeps completed intersections balanced if a cap interrupts.
        for seed in [0, 81]:
            for arm in ["acquired", "interpolation", "learned"]:
                start = time.monotonic()
                c = contracts[arm]
                if start + 100 > deadline:
                    fits.append(dict(arm=arm, seed=seed, failure="total CPU budget preflight stop"))
                    write(case / "fits.json", fits)
                    continue
                signal.alarm(100)
                try:
                    fit, _ = fit_control(
                        c["y"],
                        c["angles"],
                        uv,
                        perturb(initial[arm], seed, cfg),
                        c["reference"],
                        cfg["noise_sigma"],
                        cfg,
                        False,
                    )
                    numerical = assess(
                        fit.x, c["y"], c["angles"], uv, c["reference"], cfg, thresholds
                    )
                    result = dict(
                        arm=arm,
                        seed=seed,
                        parameters=fit.x.tolist(),
                        objective=float(2 * fit.cost),
                        converged=bool(fit.success),
                        status=int(fit.status),
                        nfev=int(fit.nfev),
                        active_indices=np.flatnonzero(fit.active_mask).tolist(),
                        numerical=numerical,
                        elapsed_seconds=time.monotonic() - start,
                    )
                except (TimeoutError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    result = dict(
                        arm=arm,
                        seed=seed,
                        failure=str(exc),
                        elapsed_seconds=time.monotonic() - start,
                    )
                finally:
                    signal.alarm(0)
                fits.append(result)
                write(case / "fits.json", fits)
                print(
                    json.dumps(
                        {k: v for k, v in result.items() if k not in ["parameters", "numerical"]}
                    ),
                    flush=True,
                )
        intersection = common_starts(fits, ["acquired", "interpolation", "learned"])
        rows = []
        predictions = {}
        for arm in ["acquired", "interpolation", "learned"]:
            chosen, gate = select_fit(
                [r for r in fits if r["arm"] == arm and r["seed"] in intersection],
                contracts[arm]["reference"],
                cfg,
                angles[mask],
                uv,
            )
            init = projection(initial[arm], angles, uv)[0].reshape(truth.shape)
            values = {
                "completion": dict(acquired=observed, interpolation=interp, learned=learned)[arm],
                "initialization": init,
            }
            if chosen is not None:
                values["refined"] = projection(np.array(chosen["parameters"]), angles, uv)[
                    0
                ].reshape(truth.shape)
                predictions[arm] = values["refined"]
            errors = {
                stage: {
                    key: float(np.linalg.norm((p - truth)[m]) / np.linalg.norm(truth[m]))
                    for key, m in [("acquired", mask), ("withheld", ~mask)]
                }
                for stage, p in values.items()
            }
            rows.append(
                dict(
                    arm=arm,
                    selected_seed=chosen["seed"] if chosen else None,
                    objective=chosen["objective"] if chosen else None,
                    errors=errors,
                    gate=gate,
                    acceptance=acceptance_parts(
                        chosen, contracts[arm]["reference"], cfg, thresholds
                    )
                    if chosen
                    else None,
                    accepted_fit_count=sum(
                        r.get("numerical", {}).get("accepted", False)
                        for r in fits
                        if r["arm"] == arm
                    ),
                    attempted_fits=2,
                )
            )
        np.savez_compressed(case / "predictions.npz", **predictions)
        report = dict(
            file=name,
            rows=rows,
            common_starts=intersection,
            initializers=info,
            failures=[
                {k: v for k, v in r.items() if k != "parameters"}
                for r in fits
                if "failure" in r or not r.get("numerical", {}).get("accepted", False)
            ],
            artifacts_sha256={p.name: sha256(p) for p in case.iterdir()},
        )
        write(case / "report.json", report)
        reports.append(report)
    write(
        folder / "report.json",
        dict(
            cases=reports,
            elapsed_seconds=time.monotonic() - begun,
            historical_seconds=historical["elapsed_seconds"],
            budget_seconds=1800,
            selected_arm=selected,
            scope="selection-only exploratory reconstruction; accepted refinement requires all numerical and multistart gates",
            input_protocol_sha256=sha256(folder / "integration_protocol.json"),
        ),
    )


if __name__ == "__main__":
    main()
