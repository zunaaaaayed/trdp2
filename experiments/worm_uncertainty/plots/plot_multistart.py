"""Plot audited multistart results, keeping calibration and stability distinct."""

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
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.input / "report.json").read_text())
    audit = json.loads((args.input / "stability_gate_audit.json").read_text())
    if (
        sha256(args.input / "report.json") != audit["source_report_sha256"]
        or sha256(args.input / "draws.json") != report["draws_sha256"]
    ):
        raise ValueError("provenance mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = audit["rows"]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for i, (domain, label) in enumerate(
        [
            ("voxel", "Voxel"),
            ("withheld_latent", "Withheld latent"),
            ("withheld_predictive", "Withheld predictive"),
        ]
    ):
        axes[0].bar(
            x + (i - 1) * 0.25,
            [r["selected_metrics"][domain]["coverage95"] * 100 for r in rows],
            0.25,
            label=label,
        )
    axes[0].axhline(95, color="black", ls="--")
    axes[0].set(title="Coverage: target ≈95%", ylabel="Coverage (%)", ylim=(0, 105))
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].bar(x, [r["selected_metrics"]["voxel"]["mean_width95"] for r in rows])
    axes[1].set(title="Voxel width: judge with coverage", ylabel="Synthetic density units")
    axes[2].bar(x, [r["selected_metrics"]["voxel"]["rmse"] for r in rows])
    axes[2].set(title="Voxel RMSE: lower is better", ylabel="Synthetic density units")
    for ax in axes:
        ax.set_xticks(x, ["Overlap\n90°/0°", "Overlap\n90°/120°", "New phantom\n90°/60°"])
    passes = sum(r["passes"] for r in rows)
    total = sum(r["draws"] for r in rows)
    fig.suptitle(
        f"Cholesky covariance, fixed shared prior, four starts: {passes}/{total} stability checks pass\nSelection uses acquired data + prior only; successful stability does not establish global optimality"
    )
    fig.savefig(args.output / "selected_results.png", dpi=150)
    plt.close(fig)
    (args.output / "results.json").write_text(
        json.dumps(dict(report=report, audit=audit, plot_code_sha256=sha256(__file__)), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
