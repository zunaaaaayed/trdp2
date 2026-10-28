"""Single predeclared validation case, completion initialization only, matched refinement."""

import argparse
import json
import resource
import time
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.data.prepare import sha256
from experiments.worm_uncertainty.data.matched_completion import evaluation_masks
from experiments.worm_uncertainty.layout import resource_path
from experiments.worm_uncertainty.core.completion_initialization import initialize_arms
from experiments.worm_uncertainty.core.cholesky_gaussian import fit_control, projection
from experiments.worm_uncertainty.core.measured_initializer import perturb
from experiments.worm_uncertainty.core.scaled_acceptance import assess
from experiments.worm_uncertainty.studies.worm_fit_controls import stability


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent", type=Path, default=Path("output/worm_uncertainty/completion_matched_v2")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    parent = json.loads((args.parent / "report.json").read_text())
    protocol = json.loads((args.parent / "protocol.json").read_text())
    selection = parent["selection"]
    cfg_design = protocol["integration"]
    if not selection["integration_approved_by_frozen_rule"]:
        raise ValueError("validation progression gate failed")
    if not args.execute:
        print(json.dumps(cfg_design, indent=2))
        return
    if args.output.exists():
        raise ValueError("output exists")
    for name, h in parent["artifacts_sha256"].items():
        if sha256(args.parent / name) != h:
            raise ValueError("parent artifact changed: " + name)
    data = np.load(args.parent / "inputs.npz")
    results = np.load(args.parent / "familiar.npz")
    fi = protocol["subsets"]["validation"].index(cfg_design["file"])
    mi = protocol["familiar_masks"].index([cfg_design["start"], cfg_design["span"]])
    indices = np.flatnonzero(
        (results["identities"][:, 0] == fi) & (results["identities"][:, 1] == mi)
    )
    indices = indices[np.argsort(results["identities"][indices, 2])]
    if len(indices) != 8:
        raise ValueError("expected all eight planes")
    measured = results["measured"][indices].transpose(1, 0, 2)
    clean = results["target"][indices].transpose(1, 0, 2)
    interp = results["interpolation"][indices].transpose(1, 0, 2)
    learned = results[selection["selected_arm"]][indices].transpose(1, 0, 2)
    mask = evaluation_masks(results, protocol, protocol["familiar_masks"])[indices[0]]
    angles = np.deg2rad(data["angles_degrees"])
    uv, v = data["uv"], data["v"]
    cfg = json.loads(resource_path("worm_shape_protocol_v1.json").read_text())
    cfg.update(components=16, budget_nfev=450, noise_sigma=protocol["noise_sd"])
    dictionary = json.loads(resource_path("worm_fit_controls_protocol_v1.json").read_text())
    thresholds = json.loads(resource_path("scaled_acceptance_protocol_v1.json").read_text())[
        "thresholds"
    ]
    args.output.mkdir(parents=True)
    started = time.monotonic()
    write(
        args.output / "protocol.json",
        dict(
            parent_sha256=sha256(args.parent / "report.json"),
            selection_sha256=sha256(args.parent / "selection.json"),
            integration=cfg_design,
            configuration=cfg,
            dictionary=dictionary,
            thresholds=thresholds,
            sources={str(p): sha256(p) for p in Path("experiments/worm_uncertainty").rglob("*.py")},
            mask_recovered_from_protocol="acquired_mask" not in results,
            paired_order="Hungarian centre matching to acquired-only reference, shared perturbation seeds; different initial scenes remain distinct",
        ),
    )
    initial, infos, contracts = initialize_arms(
        measured, interp, learned, mask, angles, uv, v, cfg, dictionary
    )
    arrays = dict(
        measured=measured,
        mask=mask,
        angles=angles,
        uv=uv,
        reference=contracts["acquired"]["reference"],
        target=clean,
        interpolation=interp,
        learned=learned,
    )
    for arm, c in contracts.items():
        arrays[arm + "_initial"] = initial[arm]
        arrays[arm + "_observations"] = c["y"]
    np.savez_compressed(args.output / "inputs.npz", **arrays)
    fits = []
    rows = []
    predictions = {}
    last_fit_seconds = 0.0
    budget_stop = False
    for arm in ["acquired", "interpolation", "learned"]:
        c = contracts[arm]
        arm_fits = []
        for seed in cfg_design["seeds"]:
            if time.monotonic() - started + last_fit_seconds > cfg_design["cpu_seconds_cap"]:
                budget_stop = True
                break
            before = time.monotonic()
            fit, _ = fit_control(
                c["y"],
                c["angles"],
                c["uv"],
                perturb(initial[arm], seed, c["cfg"]),
                c["reference"],
                c["cfg"]["noise_sigma"],
                c["cfg"],
                False,
            )
            last_fit_seconds = time.monotonic() - before
            numerical = assess(
                fit.x, c["y"], c["angles"], c["uv"], c["reference"], c["cfg"], thresholds
            )
            row = dict(
                arm=arm,
                seed=seed,
                parameters=fit.x.tolist(),
                objective=float(2 * fit.cost),
                converged=bool(fit.success),
                status=int(fit.status),
                nfev=int(fit.nfev),
                active_indices=np.flatnonzero(fit.active_mask).tolist(),
                elapsed_seconds=last_fit_seconds,
                numerical=numerical,
            )
            arm_fits.append(row)
            fits.append(row)
            write(args.output / "fits.json", fits)
            if len(fits) == 1:
                write(
                    args.output / "runtime_estimate.json",
                    dict(
                        first_fit_seconds=last_fit_seconds,
                        estimated_total_fit_seconds=12 * last_fit_seconds,
                        initialization_seconds=before - started,
                        budget_seconds=cfg_design["cpu_seconds_cap"],
                    ),
                )
            print(
                json.dumps({k: v for k, v in row.items() if k not in ["parameters", "numerical"]}),
                flush=True,
            )
        if arm_fits:
            chosen, gate = stability(arm_fits, c["angles"], c["uv"], cfg["stability_gate"])
            pred = projection(np.array(chosen["parameters"]), angles, uv)[0].reshape(clean.shape)
            predictions[arm] = pred
            metrics = {
                key: float(np.linalg.norm((pred - clean)[m]) / np.linalg.norm(clean[m]))
                for key, m in [("acquired", mask), ("withheld", ~mask)]
            }
            rows.append(
                dict(
                    arm=arm,
                    metrics=metrics,
                    gate=gate,
                    numerical=chosen["numerical"],
                    selected_seed=chosen["seed"],
                    completed_fits=len(arm_fits),
                )
            )
        if budget_stop:
            break
    np.savez_compressed(args.output / "predictions.npz", **predictions)
    before_metrics = {
        arm: float(np.linalg.norm((a - clean)[~mask]) / np.linalg.norm(clean[~mask]))
        for arm, a in [("interpolation", interp), ("learned", learned)]
    }
    report = dict(
        rows=rows,
        completion_before=before_metrics,
        initializers=infos,
        budget_stop=budget_stop,
        elapsed_seconds=time.monotonic() - started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        claim="one familiar validation case; selection by acquired objective; failed stability/local checks retained; no uncertainty claim",
        artifacts_sha256={p.name: sha256(p) for p in args.output.iterdir() if p.is_file()},
    )
    write(args.output / "report.json", report)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
