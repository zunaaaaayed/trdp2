"""Descriptive seed/deformation summaries; no raywise inference or population claims."""

import argparse
import json
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.studies.acquisition_diversity import write


def strata(results, key):
    rows = []
    for result in results:
        cases = result["metrics"]["per_case"]
        for value in sorted({r[key] for r in cases}):
            subset = [r for r in cases if r[key] == value]
            names = sorted({r["file"] for r in subset})
            per_file = {
                name: float(np.mean([r["relative_l2"] for r in subset if r["file"] == name]))
                for name in names
            }
            rows.append(
                dict(
                    arm=result["arm"],
                    seed=result.get("seed"),
                    stratum=key,
                    value=value,
                    mean=float(np.mean(list(per_file.values()))),
                    median=float(np.median(list(per_file.values()))),
                    worst=float(max(per_file.values())),
                    per_file=per_file,
                    mse=float(np.mean([r["mse"] for r in subset])),
                    boundary_mse=float(
                        np.mean(
                            [r["boundary_mse"] for r in subset if r["boundary_mse"] is not None]
                        )
                    ),
                    center_mse=float(
                        np.mean([r["center_mse"] for r in subset if r["center_mse"] is not None])
                    )
                    if any(r["center_mse"] is not None for r in subset)
                    else None,
                )
            )
    return rows


def contrasts(results, group):
    out = []
    seeds = sorted({r["seed"] for r in results})
    for left, right in [("B", "A"), ("B", "C")]:
        differences = []
        for seed in seeds:
            a = next(r for r in results if r["arm"] == left and r["seed"] == seed)["metrics"][
                "groups"
            ][group]
            b = next(r for r in results if r["arm"] == right and r["seed"] == seed)["metrics"][
                "groups"
            ][group]
            delta = {n: b["per_file"][n] - v for n, v in a["per_file"].items()}
            differences.append(
                dict(
                    seed=seed,
                    mean_improvement=b["mean"] - a["mean"],
                    relative_improvement=1 - a["mean"] / b["mean"],
                    files_improved=sum(v > 0 for v in delta.values()),
                    per_file=delta,
                )
            )
        out.append(
            dict(
                comparison=left + " versus " + right,
                group=group,
                seed_results=differences,
                mean_improvement=float(np.mean([r["mean_improvement"] for r in differences])),
                seed_sd=float(np.std([r["mean_improvement"] for r in differences], ddof=1)),
            )
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("output/worm_uncertainty/acquisition_diversity_v3")
    )
    args = parser.parse_args()
    dest = args.root / "descriptive_summary.json"
    if dest.exists():
        raise ValueError("summary exists")
    selection = json.loads((args.root / "primary/selection.json").read_text())
    audit = json.loads((args.root / "audit/report.json").read_text())
    sb = json.loads((args.root / "primary/baselines.json").read_text())["interpolation"]
    ab = audit["baselines"]["interpolation"]
    output = {}
    for role, results, baseline in [
        ("selection", selection["results"], sb),
        ("audit", audit["results"], ab),
    ]:
        output[role] = dict(
            contrasts=contrasts(results, "primary_narrow"),
            strata={
                key: strata(results + [dict(arm="interpolation", metrics=baseline)], key)
                for key in ["count", "span", "start"]
            },
        )
    profile = json.loads((args.root / "profile/report.json").read_text())
    unique_noise = sum(
        next(r["noise_preparation_seconds_shared"] for r in selection["results"] if r["seed"] == s)
        for s in {r["seed"] for r in selection["results"]}
    )
    charged = (
        profile["elapsed_seconds"]
        + sum(selection["arm_seconds"].values())
        + selection["preparation_seconds"]
        + unique_noise
        + audit["elapsed_seconds"]
    )
    output["resources"] = dict(
        profile_seconds=profile["elapsed_seconds"],
        primary_arm_seconds=selection["arm_seconds"],
        primary_preparation_seconds=selection["preparation_seconds"],
        noise_preparation_seconds=unique_noise,
        audit_seconds=audit["elapsed_seconds"],
        conservative_recorded_gpu_workflow_seconds=charged,
        ceiling_seconds=6000,
        max_training_peak_allocated=max(r["peak_allocated_bytes"] for r in selection["results"]),
        max_training_peak_reserved=max(r["peak_reserved_bytes"] for r in selection["results"]),
        note="Phase wall times include CPU preparation/validation; not CUDA-event device utilization. Runner startup/filesystem overhead outside timers excluded.",
    )
    if (args.root / "refinement/report.json").exists():
        r = json.loads((args.root / "refinement/report.json").read_text())
        output["resources"]["gaussian_cpu_wall_seconds"] = (
            r.get("elapsed_seconds", 0)
            + json.loads((args.root / "historical_completion/report.json").read_text())[
                "elapsed_seconds"
            ]
        )
    if audit["risk"] is not None:
        risk = audit["risk"]
        output["risk"] = dict(
            mean_spearman=float(np.mean([r["spearman"] for r in risk])),
            median_spearman=float(np.median([r["spearman"] for r in risk])),
            curve_means={
                k: np.mean([r[k] for r in risk], axis=0).tolist()
                for k in ["disagreement", "angular_distance", "random"]
            },
            files_disagreement_beats_geometry_at_half=sum(
                r["disagreement"][2] < r["angular_distance"][2] for r in risk
            ),
            high_error_low_disagreement_fraction_mean=float(
                np.mean([r["high_error_low_disagreement_fraction"] for r in risk])
            ),
        )
    write(dest, output)
    print(
        json.dumps(
            dict(
                resources=output["resources"],
                selection=output["selection"]["contrasts"],
                audit=output["audit"]["contrasts"],
                risk=output.get("risk"),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
