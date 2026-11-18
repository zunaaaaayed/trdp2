"""Plot separate-phantom coverage, interval width and error from verified results."""

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
    if sha256(args.input / "draws.json") != report["draws_sha256"]:
        raise ValueError("draw-level checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = report["protocol"]
    rows = report["rows"]
    groups = [(p["name"], c) for p in cfg["phantoms"] for c in cfg["conditions"]]
    acquisitions = cfg["acquisitions"]
    panels = [
        ("density", "voxel", "coverage95", "Density-only voxel coverage (%)", 100),
        ("joint", "voxel", "coverage95", "Joint voxel coverage (%)", 100),
        ("joint", "withheld_latent", "coverage95", "Joint withheld latent coverage (%)", 100),
        (
            "joint",
            "withheld_predictive",
            "coverage95",
            "Joint withheld predictive coverage (%)",
            100,
        ),
        ("joint", "voxel", "mean_width95", "Joint voxel interval width", 1),
        ("joint", "voxel", "rmse", "Voxel RMSE: lower is better", 1),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    for ax, (method, domain, metric, title, factor) in zip(axes.flat, panels):
        data = np.full((len(groups), len(acquisitions)), np.nan)
        for i, (phantom, condition) in enumerate(groups):
            for j, a in enumerate(acquisitions):
                row = next(
                    r
                    for r in rows
                    if r["phantom"] == phantom
                    and r["condition"] == condition
                    and r["span"] == a["span"]
                    and r["start"] == a["start"]
                )
                if domain in row["methods"][method]:
                    data[i, j] = row["methods"][method][domain][metric] * factor
        im = ax.imshow(
            data,
            aspect="auto",
            vmin=0,
            vmax=100 if factor == 100 else None,
            cmap="viridis" if factor == 100 else "magma",
        )
        for i in range(len(groups)):
            for j in range(len(acquisitions)):
                value = data[i, j]
                label = (
                    "N/A"
                    if np.isnan(value)
                    else (f"{value:.1f}" if factor == 100 else f"{value:.4f}")
                )
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black"
                    if np.isnan(value) or value > (100 if factor == 100 else np.nanmax(data)) * 0.6
                    else "white",
                )
        ax.set_xticks(
            range(len(acquisitions)), [f"{a['span']}° / {a['start']}°" for a in acquisitions]
        )
        ax.set_yticks(
            range(len(groups)),
            [f"{p}\n{'matched' if c == 'matched' else 'missing component'}" for p, c in groups],
        )
        ax.set_xlabel("Acquired span / start angle")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(
        "Separate 3D phantoms: mean across 10 paired noise draws × 2 starts\nCoverage target ≈95%; widths useful only with adequate coverage; units are synthetic",
        fontsize=13,
    )
    fig.savefig(args.output / "coverage_error.png", dpi=150)
    plt.close(fig)
    (args.output / "results.json").write_text(
        json.dumps(
            dict(
                report=report,
                source_report_sha256=sha256(args.input / "report.json"),
                plot_code_sha256=sha256(__file__),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
