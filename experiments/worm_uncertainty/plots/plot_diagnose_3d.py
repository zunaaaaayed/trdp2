"""Render the frozen known-3D diagnosis with explicit scale interpretation."""

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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.input / "report.json").read_text())
    if sha256(args.input / "arrays.npz") != report["arrays_sha256"]:
        raise ValueError("array checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = report["uncertainty"]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for offset, (method, label) in enumerate(
        [
            ("oracle_density", "Known centres"),
            ("learned_density", "Fitted centres: density only"),
            ("learned_joint", "Fitted centres: joint covariance"),
        ]
    ):
        axes[0].bar(
            x + (offset - 1) * 0.25,
            [r["methods"][method]["coverage95"] * 100 for r in rows],
            0.25,
            label=label,
        )
        axes[1].bar(
            x + (offset - 1) * 0.25, [r["methods"][method]["mean_width95"] for r in rows], 0.25
        )
    axes[0].axhline(95, color="black", linestyle="--", label="Nominal 95% target")
    axes[0].set(
        ylabel="Foreground voxel coverage (%)",
        ylim=(0, 105),
        title="Near 95% is the target; low means overconfident",
    )
    axes[1].set(
        ylabel="Mean 95% interval width (synthetic density)",
        title="Narrow is useful only with adequate coverage",
    )
    for ax in axes:
        ax.set_xticks(x, [str(r["span"]) + "°" for r in rows])
        ax.set_xlabel("Acquired angular span")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Known 3D phantom: mean over 30 noise draws; one fixed truth")
    fig.savefig(args.output / "coverage_width.png", dpi=160)
    plt.close(fig)
    arrays = np.load(args.input / "arrays.npz")
    z = arrays["truth"].shape[2] // 2
    truth = arrays["truth"][:, :, z].T
    mean = arrays["mean_90"][:, :, z].T
    error = abs(mean - truth)
    sd = arrays["conditional_sd_90"][:, :, z].T
    joint = arrays["joint_sd_90"][:, :, z].T
    fig, axes = plt.subplots(1, 5, figsize=(15, 4), constrained_layout=True)
    for ax, data, title, vmax, cmap in zip(
        axes,
        [truth, mean, error, sd, joint],
        [
            "Truth: brightness is density",
            "Reconstruction: compare to truth",
            "Absolute error: lower is better",
            "Density-only SD",
            "Joint centre+density SD",
        ],
        [truth.max(), truth.max(), error.max(), joint.max(), joint.max()],
        ["viridis", "viridis", "magma", "magma", "magma"],
    ):
        im = ax.imshow(data, origin="lower", extent=(-1, 1, -1, 1), vmin=0, vmax=vmax, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (normalized)")
        fig.colorbar(
            im, ax=ax, shrink=0.65, orientation="horizontal", label="Synthetic density units"
        )
    axes[0].set_ylabel("y (normalized)")
    fig.suptitle(
        "90°; first noise draw; central z slice\nSD panels share a scale: brighter means less certain; dark is not proof of accuracy",
        fontsize=11,
    )
    fig.savefig(args.output / "slices.png", dpi=160)
    plt.close(fig)
    provenance = dict(
        report=report,
        source_report_sha256=sha256(args.input / "report.json"),
        plot_code_sha256=sha256(__file__),
    )
    (args.output / "results.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
