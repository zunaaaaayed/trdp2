"""Plot failed importance correction and mixing honestly; do not imply calibration."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.worm_uncertainty.data.prepare import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("output/worm_uncertainty/nonlinear_posterior_v1")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    args.output.mkdir(parents=True)
    report = json.loads((args.source / "report.json").read_text())
    ref = json.loads((args.source / "reference.json").read_text())
    labels = ["Moderate\n14.4°", "Moderate\n72°", "Elongated\n14.4°", "Elongated\n72°"]
    joint = [r["methods"]["joint"] for r in report["rows"]]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    x = np.arange(4)
    ax[0, 0].bar(x, [r["importance"]["ess"] for r in joint], color="#c44e52")
    ax[0, 0].axhline(400, color="black", ls="--", label="Required ESS ≥400")
    ax[0, 0].set(
        yscale="log",
        ylabel="ESS out of 2,048 draws (log scale)",
        title="All joint corrections fail adequacy",
    )
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].bar(x, [100 * r["importance"]["maximum_weight"] for r in joint], color="#c44e52")
    ax[0, 1].axhline(2, color="black", ls="--", label="Required maximum ≤2%")
    ax[0, 1].set(ylabel="Largest normalized weight (%)", title="Lower concentration is better")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].bar(x, [100 * (1 - r["valid_fraction"]) for r in joint], color="#8172b3")
    ax[1, 0].set(ylabel="Out-of-bound draws (%)", title="Zero violations do not establish validity")
    ax[1, 1].bar(
        x, [r["nonlinear_valid_discrepancy_sigma_quantiles"][0] for r in joint], color="#4c72b0"
    )
    ax[1, 1].set(
        ylabel="Median nonlinear − linear RMS / noise SD",
        title="Valid draws only; smaller is more local",
    )
    for a in ax.ravel():
        a.set_xticks(x, labels)
    fig.savefig(args.output / "importance_diagnostics.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    methods = ["local_nonlinear", "truncated_gaussian", "weighted"]
    names = ["Joint Gaussian", "Bound-valid subset", "Importance (FAILED)"]
    for i, (method, label) in enumerate(zip(methods, names)):
        width = 0.25
        axes[0].bar(
            x + (i - 1) * width,
            [100 * r[method]["withheld"]["coverage"] for r in joint],
            width,
            label=label,
        )
        axes[1].bar(x + (i - 1) * width, [r[method]["withheld"]["width"] for r in joint], width)
        axes[2].bar(
            x + (i - 1) * width, [r[method]["withheld"]["interval_score"] for r in joint], width
        )
    axes[0].axhline(95, color="black", ls="--")
    axes[0].legend(fontsize=7)
    axes[0].set(
        ylabel="Withheld latent coverage (%)", ylim=(0, 105), title="95% target; four cases only"
    )
    axes[1].set(ylabel="Mean interval width (model units)", title="Narrower alone is not better")
    axes[2].set(
        ylabel="Mean 95% interval score (model units)",
        title="Proper interval score: lower is better",
    )
    for a in axes:
        a.set_xticks(x, labels)
    fig.suptitle(
        "Failed importance summaries are diagnostic only; truncation is not exact inference",
        fontsize=11,
    )
    fig.savefig(args.output / "withheld_diagnostics.png", dpi=180)
    plt.close(fig)
    summary = dict(
        audit=report,
        reference=ref,
        source_artifacts_sha256={p.name: sha256(p) for p in args.source.iterdir() if p.is_file()},
        plot_code_sha256=sha256(Path(__file__)),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
