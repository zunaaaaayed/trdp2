"""Display frozen local-acceptance validation separately from density accuracy."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from prepare import sha256
from cholesky_gaussian import projection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    report = json.loads((args.input / "report.json").read_text())
    for n, h in report["artifacts_sha256"].items():
        if sha256(args.input / n) != h:
            raise ValueError("artifact mismatch: " + n)
    cases = report["cases"]
    if len(cases) != 32 or len({c["case"] for c in cases}) != 32:
        raise ValueError("32 cases required")
    fits = json.loads((args.input / "fits.json").read_text())
    inputs = np.load(args.input / "inputs.npz")
    multistart = []
    thresholds = report["base_configuration"]["stability_gate"]
    for case in cases:
        pair = [
            next(f for f in fits if f["case"] == case["case"] and f["stage"] == stage)
            for stage in ["final", "oracle"]
        ]
        mask = inputs[case["case"] + "_mask"]
        predictions = [
            projection(np.array(f["parameters"]), inputs["angles"][mask], inputs["uv"])[0]
            for f in pair
        ]
        gap = abs(pair[0]["objective"] - pair[1]["objective"]) / max(
            1.0, min(f["objective"] for f in pair)
        )
        disagreement = float(
            np.linalg.norm(predictions[0] - predictions[1])
            / max(np.linalg.norm(predictions[0]), 1e-30)
        )
        eligible = all(
            f["assessment"]["accepted"] and not f["assessment"]["active_indices"] for f in pair
        )
        multistart.append(
            dict(
                case=case["case"],
                relative_objective_gap=gap,
                acquired_prediction_disagreement=disagreement,
                passed=bool(
                    eligible
                    and gap <= thresholds["relative_objective_gap"]
                    and disagreement <= thresholds["relative_prediction_disagreement"]
                ),
            )
        )
    counts = {
        stage: dict(
            scaled=sum(c["stages"][stage]["assessment"]["accepted"] for c in cases),
            original=sum(c["stages"][stage]["converged"] for c in cases),
        )
        for stage in ["initial", "partial", "final", "polish", "oracle"]
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x = np.arange(5)
    for offset, key, label in [
        (-0.18, "original", "Original absolute gradient"),
        (0.18, "scaled", "New local rule"),
    ]:
        bars = axes[0].bar(x + offset, [counts[s][key] for s in counts], width=0.35, label=label)
        axes[0].bar_label(bars, fontsize=7)
    axes[0].set(
        xticks=x,
        xticklabels=list(counts),
        ylim=(0, 37),
        ylabel="Accepted cases / 32",
        title="Counts alone do not establish a good rule",
    )
    axes[0].legend(fontsize=7)
    for i, c in enumerate(cases):
        a = c["stages"]["final"]["assessment"]["accepted"]
        color = "#228833" if a else "#AA3377"
        axes[1].scatter(
            i,
            max(c["polish_relative_objective_gain"], 1e-16),
            color=color,
            marker="o" if a else "x",
        )
        axes[2].scatter(
            i,
            max(c["stages"]["final"]["truth_density_error"] * 100, 1e-8),
            color=color,
            marker="o" if a else "x",
        )
    axes[1].axhline(
        report["protocol"]["validation"]["max_relative_objective_gain"], color="black", ls="--"
    )
    axes[1].set(
        yscale="log",
        xlabel="Fresh case index",
        ylabel="Objective gain from polishing — lower is better",
        title="Circle accepted; cross rejected\nOther stability limits also checked",
    )
    axes[2].set(
        yscale="log",
        xlabel="Fresh case index",
        ylabel="Exact density error (%) — lower is better",
        title="Numerical acceptance is not accuracy\nSame accepted/rejected markers",
    )
    fig.suptitle(
        "Frozen scale-aware local validation · oracle geometry priors · no posterior calibration claim"
    )
    fig.tight_layout()
    args.output.mkdir(parents=True)
    fig.savefig(args.output / "validation.png", dpi=160)
    plt.close(fig)
    summary = dict(
        multistart=multistart,
        counts=counts,
        decision=report["decision"],
        cases=cases,
        protocol=report["protocol"],
        report_sha256=sha256(args.input / "report.json"),
        plot_code_sha256=sha256(__file__),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(counts=counts, decision=report["decision"]), indent=2))


if __name__ == "__main__":
    main()
