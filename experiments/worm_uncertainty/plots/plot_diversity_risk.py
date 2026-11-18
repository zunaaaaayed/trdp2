"""Explicit post-audit disagreement failure and tie-order sensitivity figures."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.worm_uncertainty.data.acquisition_diversity import read_design
from experiments.worm_uncertainty.data.matched_completion import grid
from experiments.worm_uncertainty.data.prepare import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("output/worm_uncertainty/acquisition_diversity_v3")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/worm_uncertainty/acquisition_diversity_v3"),
    )
    args = parser.parse_args()
    out = args.output
    if (out / "disagreement_failure.png").exists():
        raise ValueError("risk figures exist")
    d, _ = read_design()
    r = json.loads((args.root / "risk_sensitivity.json").read_text())
    audit = json.loads((args.root / "audit/report.json").read_text())
    a = np.load(args.root / "audit/ensemble.npz")
    clean = np.load(args.root / "audit/clean.npy")
    w = r["worst_low_disagreement"]
    f, mi, zi = w["file_index"], w["mask_index"], w["slice_index"]
    truth = clean[f, zi]
    pred = a["mean"][f, mi, zi]
    sd = a["std"][f, mi, zi]
    err = abs(pred - truth)
    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    vmin = min(0, truth.min(), pred.min())
    vmax = max(truth.max(), pred.max())
    for i, (value, title) in enumerate(
        [
            (truth, "Clean target"),
            (pred, "B ensemble mean"),
            (err, "Absolute error: lower is better"),
            (sd, "Disagreement: not calibrated"),
        ]
    ):
        im = axes[i].imshow(
            value,
            aspect="auto",
            origin="lower",
            vmin=vmin if i < 2 else 0,
            vmax=vmax if i < 2 else None,
            cmap="viridis" if i < 2 else "inferno",
        )
        axes[i].set_title(title, fontsize=10)
        axes[i].plot(w["detector_index"], w["angle_index"], "wx", markersize=8)
        axes[i].set_xlabel("Detector sample")
        axes[i].set_ylabel("Angle sample")
        fig.colorbar(im, ax=axes[i], label="Normalized units", shrink=0.85)
    ax = fig.add_subplot(gs[1, :2])
    _, uv, _ = grid(d)
    u = np.unique(uv[:, 0])
    row = w["angle_index"]
    ax.plot(u, truth[row], color="black", label="clean target", lw=2)
    for i, seed in enumerate(d["training_seeds"]):
        model = np.load(args.root / "audit" / f"{audit['selected_arm']}_{seed}.npy", mmap_mode="r")
        ax.plot(u, model[f, mi, zi, row], alpha=0.75, label=f"seed {i + 1}")
    ax.axvline(u[w["detector_index"]], color="gray", ls=":")
    ax.set_xlabel("Detector u")
    ax.set_ylabel("Normalized projection")
    ax.legend(fontsize=8)
    ax.set_title(f"Missing angle {row * 7.2:.1f}°: models agree on an incorrect peak")
    ax = fig.add_subplot(gs[1, 2:])
    fractions = np.array(d["analysis"]["fractions"]) * 100
    for key, label, color in [
        ("disagreement_tie_expected", "Disagreement", "#2075a9"),
        ("geometry_tie_expected", "Geometry, expected tie order", "#c87922"),
        ("random_expected", "Random expectation", "#777777"),
    ]:
        ax.plot(fractions, r["curve_means"][key], marker="o", label=label, color=color)
    ax.set_xlabel("Lowest-score missing rays retained (%)")
    ax.set_ylabel("Absolute clean-ray error: lower is better")
    ax.set_title("Post-audit tie-order sensitivity (no tuning)")
    ax.legend(fontsize=8)
    fig.suptitle(
        f"High-error / low-disagreement diagnostic: {w['file']}, z{w['z']}, acquired {w['mask']['span']}°\nSelected after audit by declared diagnostic rule; error={w['absolute_error']:.3f}, model SD={w['disagreement']:.3f}. No coverage claim.",
        fontsize=12,
    )
    fig.savefig(out / "disagreement_failure.png", dpi=160)
    plt.close(fig)
    (out / "risk_sensitivity_summary.json").write_text(
        json.dumps(
            dict(source_sha256=sha256(args.root / "risk_sensitivity.json"), report=r), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
