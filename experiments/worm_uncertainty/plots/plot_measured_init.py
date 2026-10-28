"""Display fresh-phantom measured-only initialization and stability results."""

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
    if sha256(args.input / "report.json") != audit["source_report_sha256"]:
        raise ValueError("report checksum mismatch")
    for name, ext in [("draws", "json"), ("arrays", "npz")]:
        if sha256(args.input / f"{name}.{ext}") != report[name + "_sha256"]:
            raise ValueError("artifact checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = audit["rows"]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for i, (domain, label) in enumerate(
        [
            ("voxel", "Voxel"),
            ("withheld_latent", "Withheld latent"),
            ("withheld_predictive", "Withheld predictive"),
        ]
    ):
        axes[0].bar(
            x + (i - 1) * 0.25,
            [r["selected_metrics"].get(domain, {}).get("coverage95", np.nan) * 100 for r in rows],
            0.25,
            label=label,
        )
    axes[0].axhline(95, color="black", ls="--")
    axes[0].set(title="Coverage: target ≈95%", ylabel="Coverage (%)", ylim=(0, 105))
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].bar(
        x, [r["selected_metrics"].get("voxel", {}).get("mean_width95", np.nan) for r in rows]
    )
    axes[1].set(title="Voxel width: judge with coverage", ylabel="Synthetic density units")
    labels = []
    for row in rows:
        eligible = sum("metrics" in r for r in audit["records"] if r["case"] == row["case"])
        case = next(c for c in report["protocol"]["cases"] if c["name"] == row["case"])
        phantom = row["case"].split("_")[1]
        labels.append(
            f"{phantom} {case['span']}°/{case['start']}°\n{row['passes']}/{row['draws']} gates pass; {eligible} evaluated"
        )
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=8)
    fig.suptitle(
        "Starts and prior reference use acquired data only; ten noise draws per case\nMetrics include failed stability checks when ≥2 candidates converge; no acceptance claim"
    )
    fig.savefig(args.output / "coverage.png", dpi=150)
    plt.close(fig)
    arrays = np.load(args.input / "arrays.npz")
    # Fixed case/draw/slice, specified independently of its numerical quality.
    case = "fresh_spread_90_75"
    row = next(r for r in audit["records"] if r["case"] == case and r["noise_repeat"] == 0)
    if "selected_seed" in row:
        z = report["protocol"]["voxel_shape"][2] // 2
        seed = row["selected_seed"]
        truth = arrays[case + "_truth"][:, :, z].T
        mean = arrays[f"{case}_mean_{seed}"][:, :, z].T
        sd = arrays[f"{case}_sd_{seed}"][:, :, z].T
        error = abs(mean - truth)
        fig, axes = plt.subplots(1, 4, figsize=(13, 4), constrained_layout=True)
        for ax, data, title, vmax, cmap in zip(
            axes,
            [truth, mean, error, sd],
            [
                "Truth: brightness is density",
                "Selected reconstruction",
                "Absolute error: lower is better",
                "SD: low alone is not accuracy",
            ],
            [max(truth.max(), mean.max())] * 2 + [max(error.max(), sd.max())] * 2,
            ["viridis", "viridis", "magma", "magma"],
        ):
            im = ax.imshow(
                data, origin="lower", extent=(-1, 1, -1, 1), vmin=0, vmax=vmax, cmap=cmap
            )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("x (normalized)")
            fig.colorbar(im, ax=ax, orientation="horizontal", label="Synthetic density units")
        axes[0].set_ylabel("y (normalized)")
        fig.suptitle(
            f"Spread 90°/75°; noise draw 0; central z slice; stability gate passed={row['passed']}\nTruth/mean share limits; error/SD share limits"
        )
        fig.savefig(args.output / "slices.png", dpi=150)
        plt.close(fig)
    (args.output / "results.json").write_text(
        json.dumps(dict(report=report, audit=audit, plot_code_sha256=sha256(__file__)), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
