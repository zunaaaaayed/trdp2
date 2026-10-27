"""Plot acquired-only fit selection and withheld error with explicit acceptance status."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists; use a fresh path")
    report = json.loads((args.input / "report.json").read_text())
    for name, digest in report["artifacts_sha256"].items():
        if sha256(args.input / name) != digest:
            raise ValueError("artifact mismatch: " + name)
    fits = json.loads((args.input / "fits.json").read_text())
    rows = report["rows"]
    args.output.mkdir(parents=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = ["Joint 450", "Staged 50+50+350"]
    x = np.arange(len(rows))
    for offset, key, label in [
        (-0.18, "measured_relative_l2", "Acquired"),
        (0.18, "withheld_relative_l2", "Withheld"),
    ]:
        bars = axes[0].bar(x + offset, [100 * r[key] for r in rows], width=0.35, label=label)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=8)
    axes[0].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Clean projection L2 error (%) — lower is better",
        title="Selected using acquired data only",
    )
    axes[0].legend()
    for i, row in enumerate(rows):
        candidates = [f for f in fits if f["arm"] == row["arm"]]
        for f in candidates:
            axes[1].scatter(
                i,
                f["objective"],
                marker="o" if f["converged"] else "x",
                color="#228833" if f["converged"] else "#AA3377",
            )
        gate = row["gate"]
        axes[2].bar(
            i, gate["converged_candidates"], color="#228833" if gate["passed"] else "#AA3377"
        )
        axes[2].text(
            i,
            gate["converged_candidates"] + 0.15,
            "PASS" if gate["passed"] else "FAIL",
            ha="center",
        )
    axes[1].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Shared objective — lower is better",
        title="Four starts: circle converged, cross unfinished",
    )
    axes[2].set(
        xticks=x,
        xticklabels=labels,
        ylim=(0, 5),
        yticks=range(5),
        ylabel="Converged starts / 4",
        title="Gate also checks agreement and bounds",
    )
    fig.suptitle("Development worm · 16 Gaussians · 120° · one noise draw; no calibration claim")
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=170)
    plt.close(fig)
    summary = dict(
        rows=rows,
        density_audit=report["density_audit"],
        report_sha256=sha256(args.input / "report.json"),
        elapsed_seconds=report["elapsed_seconds"],
        source_sha256=report["source_sha256"],
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
