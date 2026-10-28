"""Compare bound/prior controls with the adaptive Cholesky follow-up."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.worm_uncertainty.data.prepare import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--factorial", type=Path, required=True)
    p.add_argument("--cholesky", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    reports = {}
    for name, path in [("factorial", args.factorial), ("cholesky", args.cholesky)]:
        report = json.loads((path / "report.json").read_text())
        if sha256(path / "draws.json") != report["draws_sha256"]:
            raise ValueError("draw checksum mismatch")
        reports[name] = report
    args.output.mkdir(parents=True, exist_ok=False)
    rows = reports["factorial"]["rows"] + reports["cholesky"]["rows"]
    cases = [x["name"] for x in reports["factorial"]["protocol"]["cases"]]
    variants = [
        "bounded_start_prior",
        "unbounded_start_prior",
        "bounded_shared_prior",
        "unbounded_shared_prior",
        "cholesky_shared_prior",
    ]
    labels = [
        "Euler bounded; start prior",
        "Euler unbounded; start prior",
        "Euler bounded; shared prior",
        "Euler unbounded; shared prior",
        "Cholesky; shared prior*",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    x = np.arange(len(cases))
    for i, (variant, label) in enumerate(zip(variants, labels)):
        values = [
            next(r for r in rows if r["case"] == case and r["variant"] == variant) for case in cases
        ]
        offset = x + (i - 2) * 0.15
        axes[0, 0].bar(
            offset, [r["paired_mean_difference_rmse"] for r in values], 0.15, label=label
        )
        axes[0, 1].bar(offset, [r["metrics"]["voxel"]["coverage95"] * 100 for r in values], 0.15)
        axes[1, 0].bar(offset, [r["metrics"]["voxel"]["rmse"] for r in values], 0.15)
        axes[1, 1].bar(offset, [r["metrics"]["voxel"]["mean_width95"] for r in values], 0.15)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Difference between starts: lower is more stable")
    axes[0, 0].set_ylabel("Mean paired voxel RMSE (log scale)")
    axes[0, 1].axhline(95, color="black", ls="--")
    axes[0, 1].set(title="Voxel coverage: target ≈95%", ylabel="Coverage (%)", ylim=(0, 105))
    axes[1, 0].set(title="Reconstruction RMSE: lower is better", ylabel="Synthetic density units")
    axes[1, 1].set(
        title="Interval width: useful only with adequate coverage", ylabel="Synthetic density units"
    )
    for ax in axes.flat:
        ax.set_xticks(x, ["Overlap 90°/0°", "Overlap 90°/120°", "New phantom 90°/60°"])
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9
    )
    fig.suptitle(
        "Rotation bounds and initialization sensitivity: all-start averages over 10 paired noise draws\n*Cholesky follow-up also changes shape prior, shape bounds and solver scaling; no isolated causal claim",
        fontsize=12,
    )
    fig.savefig(args.output / "controls.png", dpi=150)
    plt.close(fig)
    result = dict(
        reports=reports,
        source_report_sha256={
            name: sha256(path / "report.json")
            for name, path in [("factorial", args.factorial), ("cholesky", args.cholesky)]
        },
        plot_code_sha256=sha256(__file__),
    )
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
