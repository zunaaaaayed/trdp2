"""Plot verified restoration and nested shape-covariance controls."""

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
    p.add_argument("--parent", type=Path, default=Path("output/worm_uncertainty/phantom_stress_v1"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    r = json.loads((args.input / "report.json").read_text())
    parent = json.loads((args.parent / "report.json").read_text())
    if sha256(args.parent / "report.json") != r["parent_report_sha256"]:
        raise ValueError("parent changed")
    for filename in ["draws", "arrays"]:
        extension = "json" if filename == "draws" else "npz"
        if sha256(args.input / f"{filename}.{extension}") != r[filename + "_sha256"]:
            raise ValueError("artifact changed")
    args.output.mkdir(parents=True, exist_ok=False)
    phantoms = [p["name"] for p in r["protocol"]["phantoms"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for i, name in enumerate(phantoms):
        restored = [
            x for x in r["rows"] if x["phase"] == "restored_component" and x["phantom"] == name
        ]
        omitted = [
            next(
                x
                for x in parent["rows"]
                if x["condition"] == "omitted_component"
                and x["phantom"] == name
                and x["span"] == row["span"]
                and x["start"] == row["start"]
            )
            for row in restored
        ]
        for values, label in [(omitted, "Missing component"), (restored, "Restored component")]:
            axes[i, 0].plot(
                [x["methods"]["joint"]["voxel"]["coverage95"] * 100 for x in values],
                "o-",
                label=label,
            )
            axes[i, 1].semilogy(
                [x["methods"]["joint"]["voxel"]["rmse"] for x in values], "o-", label=label
            )
        axes[i, 0].axhline(95, ls="--", color="black", label="95% target")
        axes[i, 0].set_ylim(0, 105)
        axes[i, 0].set_ylabel(name + "\nVoxel coverage (%)")
        axes[i, 1].set_ylabel(name + "\nVoxel RMSE (log scale)")
        for ax in axes[i]:
            ax.set_xticks(range(len(restored)), [f"{x['span']}°/{x['start']}°" for x in restored])
            ax.set_xlabel("Span / start")
            ax.legend(fontsize=8)
    axes[0, 0].set_title("Coverage should be near 95%")
    axes[0, 1].set_title("Lower RMSE is better")
    fig.suptitle(
        "Restore the fourth component: same truth, noise draws and acquisition windows\nWidths known; centre+density covariance; synthetic density units"
    )
    fig.savefig(args.output / "repair.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    methods = ["density", "centres", "widths", "rotations"]
    labels = ["Density only", "+ centres", "+ log widths", "+ rotations (all)"]
    for i, name in enumerate(phantoms):
        rows = [x for x in r["rows"] if x["phase"] == "learned_shape" and x["phantom"] == name]
        for method, label in zip(methods, labels):
            axes[i, 0].plot(
                [x["methods"][method]["voxel"]["coverage95"] * 100 for x in rows], "o-", label=label
            )
            axes[i, 1].plot(
                [
                    x["methods"][method].get("withheld_latent", {}).get("coverage95", np.nan) * 100
                    for x in rows
                ],
                "o-",
            )
            axes[i, 2].plot([x["methods"][method]["voxel"]["mean_width95"] for x in rows], "o-")
        for j, ax in enumerate(axes[i]):
            ax.set_xticks(range(len(rows)), [f"{x['span']}°/{x['start']}°" for x in rows])
            ax.set_xlabel("Span / start")
            if j < 2:
                ax.axhline(95, ls="--", color="black")
                ax.set_ylim(0, 105)
        axes[i, 0].set_ylabel(name + "\nCoverage (%)")
        axes[i, 2].set_ylabel("Synthetic density units")
        axes[i, 0].legend(fontsize=8, loc="lower left")
    for ax, title in zip(
        axes[0],
        [
            "Voxel coverage: target ≈95%",
            "Withheld latent coverage: target ≈95%",
            "Voxel width: judge with coverage",
        ],
    ):
        ax.set_title(title)
    shape_rows = [row for row in r["rows"] if row["phase"] == "learned_shape"]
    active = sum(row["active_bounds"] for row in shape_rows)
    total = sum(row["fits"] for row in shape_rows)
    repeats = r["protocol"]["noise_repeats"]
    starts = len(r["protocol"]["initialization_seeds"])
    fig.suptitle(
        "Rotated four-component truth; all means identical within each fitted case\n"
        f"{repeats} noise draws × {starts} starts; {active}/{total} fits reach parameter bounds "
        "(covariance ignores truncation)"
    )
    fig.savefig(args.output / "shape_covariance.png", dpi=150)
    plt.close(fig)
    (args.output / "results.json").write_text(
        json.dumps(
            dict(
                report=r,
                parent_rows=parent["rows"],
                source_report_sha256=sha256(args.input / "report.json"),
                plot_code_sha256=sha256(__file__),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
