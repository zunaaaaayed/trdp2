"""Plot completed continuation diagnostics without changing failed acceptance gates."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from prepare import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--numerical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    report = json.loads((args.input / "report.json").read_text())
    for n, h in report["artifacts_sha256"].items():
        if sha256(args.input / n) != h:
            raise ValueError("artifact hash mismatch")
    numerical = json.loads((args.numerical / "report.json").read_text())
    if numerical["parent_report_sha256"] != sha256(args.input / "report.json"):
        raise ValueError("numerical parent mismatch")
    fits = json.loads((args.input / "fits.json").read_text())
    labels = [("J" if f["arm"] == "joint_450" else "S") + str(f["seed"]) for f in fits]
    x = np.arange(len(fits))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for key, label in [("before", "Before"), ("after", "After")]:
        axes[0].plot(x, [f[key]["projected_gradient_inf"] for f in fits], "o-", label=label)
    axes[0].axhline(1e-6, color="red", ls="--", label="Frozen target")
    axes[0].set(
        yscale="log",
        ylabel="Projected-gradient norm — lower is better",
        title="0/8 pass stationarity",
    )
    axes[0].legend(fontsize=8)
    improvements = [
        100 * (f["before"]["objective"] - f["after"]["objective"]) / f["before"]["objective"]
        for f in fits
    ]
    bars = axes[1].bar(x, improvements)
    axes[1].bar_label(bars, fmt="%.2f", fontsize=8)
    axes[1].set(
        ylabel="Objective reduction (%) — larger is better",
        title="Selected solutions barely change",
        ylim=(0, 18),
    )
    axes[2].bar(x, [f["prior_scaled_jacobian_condition"] for f in numerical["rows"]])
    axes[2].set(
        ylabel="Jacobian condition number (prior-scaled)",
        title="Larger means more unequal local sensitivities\nNo universal acceptance threshold",
    )
    for ax in axes:
        ax.set_xticks(x, labels, rotation=45)
    fig.suptitle("Acquired-only continuation · J = joint, S = staged · no posterior claim")
    fig.tight_layout()
    args.output.mkdir(parents=True)
    fig.savefig(args.output / "continuation.png", dpi=160)
    plt.close(fig)
    summary = dict(
        rows=report["rows"],
        fits=[{k: v for k, v in f.items() if k != "parameters"} for f in fits],
        numerical=numerical,
        report_sha256=sha256(args.input / "report.json"),
        numerical_report_sha256=sha256(args.numerical / "report.json"),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
