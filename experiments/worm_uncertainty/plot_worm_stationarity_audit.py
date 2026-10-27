"""Plot acquired residuals and geometry diagnostics before continuation."""

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
        raise ValueError("output exists")
    rows = json.loads((args.input / "initial_audit.json").read_text())
    provenance = json.loads((args.input / "provenance.json").read_text())
    labels = [("J" if r["arm"] == "joint_450" else "S") + str(r["seed"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for r, label in zip(rows, labels):
        axes[0].plot(r["residual_rms_sigma_by_angle"], label=label, alpha=0.8)
    axes[0].axhline(1, color="black", ls=":", label="Noise RMS reference")
    axes[0].set(
        xlabel="Acquired sampled angle index",
        ylabel="Residual RMS / noise sigma",
        title="Residuals exceed noise scale\nLower is better; 1 is a reference, not a test",
    )
    axes[0].legend(ncol=3, fontsize=7)
    x = np.arange(len(rows))
    for i, r in enumerate(rows):
        widths = np.array(r["principal_widths"])
        axes[1].scatter(np.full(widths.size, i), widths.ravel(), s=10, alpha=0.6)
    axes[1].set(
        xticks=x,
        xticklabels=labels,
        yscale="log",
        ylabel="Principal Gaussian SD (model coordinates)",
        title="All three widths of all 16 components\nSpread is diagnostic; no good/bad cutoff",
    )
    axes[2].bar(x, [r["projected_gradient_inf"] for r in rows])
    axes[2].axhline(provenance["protocol"]["projected_gradient_tolerance"], color="red", ls="--")
    axes[2].set(
        xticks=x,
        xticklabels=labels,
        yscale="log",
        ylabel="Projected-gradient infinity norm",
        title="Lower is better: target ≤ 1e−6\nAll saved starts fail this stricter check",
    )
    for ax in axes[1:]:
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Before continuation: J = joint, S = staged · acquired data only")
    fig.tight_layout()
    args.output.mkdir(parents=True)
    fig.savefig(args.output / "initial_audit.png", dpi=160)
    plt.close(fig)
    (args.output / "summary.json").write_text(
        json.dumps(
            dict(
                rows=rows,
                provenance=provenance,
                initial_audit_sha256=sha256(args.input / "initial_audit.json"),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
