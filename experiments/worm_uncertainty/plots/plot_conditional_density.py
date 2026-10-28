"""Render verified limited-angle and density-posterior results with reading guides."""

import argparse
import json
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.studies.check_fbp import plt
from experiments.worm_uncertainty.data.prepare import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--posterior", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    br = json.loads((args.baseline / "report.json").read_text())
    pr = json.loads((args.posterior / "report.json").read_text())
    if sha256(args.baseline / "report.json") != pr["baseline_report_sha256"]:
        raise ValueError("baseline provenance mismatch")
    for row in br["results"]:
        for name, h in row["artifacts"].items():
            if sha256(args.baseline / row["case"] / name) != h:
                raise ValueError("baseline artifact mismatch")
    for row in pr["results"]:
        if sha256(args.posterior / row["case"] / "arrays.npz") != row["arrays_sha256"]:
            raise ValueError("posterior artifact mismatch")
    spans = [180, 120, 90]
    images = []
    for span in spans:
        clean = np.load(args.baseline / f"span{span}_sigma0" / "arrays.npz")["volume_xyz"][
            105:205, 105:205, 4
        ].T
        noisy = np.load(args.baseline / f"span{span}_sigma0.05" / "arrays.npz")["volume_xyz"][
            105:205, 105:205, 4
        ].T
        post = np.load(args.posterior / f"span{span}_sigma0.05" / "arrays.npz")
        images.append([clean, noisy, post["voxel_mean"], post["voxel_sd"]])
    lo = min(v.min() for row in images for v in row[:3])
    hi = max(v.max() for row in images for v in row[:3])
    sdmax = max(row[3].max() for row in images)
    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
    titles = [
        "Native fit, clean",
        "Native fit, 5% noise",
        "Conditional density mean",
        "Conditional SD\nhigher = less certain",
    ]
    for i, span in enumerate(spans):
        for j in range(4):
            im = axes[i, j].imshow(
                images[i][j],
                vmin=0 if j == 3 else lo,
                vmax=sdmax if j == 3 else hi,
                cmap="viridis" if j == 3 else "gray",
            )
            axes[i, j].set_title(f"{span}°: {titles[j]}", fontsize=9)
            axes[i, j].set_axis_off()
            if i == 2:
                fig.colorbar(
                    im, ax=axes[:, j].tolist(), shrink=0.55, label="uncalibrated density units"
                )
    fig.suptitle(
        "Same slice and crop; brightness is not accuracy. No voxel ground truth.\nWARNING: density-only intervals fail withheld projection calibration; low SD can be misleading.",
        fontsize=11,
    )
    fig.savefig(args.output / "slab_comparison.png", dpi=140)
    plt.close(fig)
    selected = [row for row in pr["results"] if row["span"] < 180]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = np.arange(2)
    for offset, key, label in [
        (-0.18, "withheld_latent_clean", "Latent interval vs clean"),
        (0.18, "withheld_predictive_noisy", "Predictive interval vs noisy"),
    ]:
        axes[0].bar(
            x + offset,
            [100 * r["coverage"][key]["coverage95"] for r in selected],
            width=0.36,
            label=label,
        )
        axes[1].bar(
            x + offset,
            [r["coverage"][key]["mean_width95"] for r in selected],
            width=0.36,
            label=label,
        )
    axes[0].axhline(95, color="black", linestyle="--", label="Nominal target: 95%")
    axes[0].set(
        ylim=(0, 101),
        ylabel="Foreground coverage (%)",
        title="Severe undercoverage = overconfidence",
    )
    axes[1].set(
        ylabel="Width in normalized projection units",
        title="Narrower helps only with adequate coverage",
    )
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(["120°", "90°"])
    axes[0].legend(fontsize=8)
    fig.savefig(args.output / "coverage.png", dpi=140)
    plt.close(fig)
    inputs = np.load(args.baseline / "inputs.npz")
    cfg = json.loads((args.baseline / "protocol.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, span in zip(axes, [120, 90]):
        post = np.load(args.posterior / f"span{span}_sigma0.05" / "arrays.npz")
        angle = 80
        z = 4
        mean = post["prediction"][angle, z]
        sd = np.sqrt(post["latent_sd"][angle, z] ** 2 + cfg["posterior_noise_sigma"] ** 2)
        clean = inputs["clean"][angle, z]
        noisy = clean + cfg["posterior_noise_sigma"] * inputs["unit_noise"][angle, z]
        detector = np.arange(310)
        ax.plot(detector, clean, label="Clean withheld reference", color="black")
        ax.plot(detector, noisy, label="Noisy withheld measurements", color="gray", alpha=0.5)
        ax.plot(detector, mean, label="Conditional mean", color="tab:blue")
        ax.fill_between(
            detector,
            mean - 1.95996398454 * sd,
            mean + 1.95996398454 * sd,
            color="tab:blue",
            alpha=0.25,
            label="Nominal 95% predictive interval",
        )
        ax.set(
            xlim=(105, 205),
            xlabel="detector pixel",
            ylabel="normalized projection",
            title=f"{span}° fit; held-out 144° view",
        )
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Good prediction should track the reference; intervals should include about 95% of future measurements.\nVisible misses expose model error that density-only uncertainty does not capture.",
        fontsize=10,
    )
    fig.savefig(args.output / "withheld_profile.png", dpi=140)
    plt.close(fig)
    record = dict(
        baseline=br,
        posterior=pr,
        renderer_sha256=sha256(__file__),
        crop=[105, 205, 105, 205],
        source_z=256,
        profile_angle_deg=144,
        mean_scale=[float(lo), float(hi)],
        sd_scale=[0, float(sdmax)],
        figures={
            n: sha256(args.output / n)
            for n in ["slab_comparison.png", "coverage.png", "withheld_profile.png"]
        },
    )
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
