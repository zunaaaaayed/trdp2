"""Build compact Week 6 report figures from frozen results; no experiment reruns."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from experiments.worm_uncertainty.core.completion_evaluation import (
    filtered_backprojection,
    polar_spectrum,
    density_slice,
)
from experiments.worm_uncertainty.data.acquisition_diversity import read_design
from experiments.worm_uncertainty.data.matched_completion import grid
from experiments.worm_uncertainty.data.prepare import sha256

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "Reports/Week6/figures"
DATA = ROOT / "docs/figures/worm_uncertainty"
RUN = ROOT / "output/worm_uncertainty/acquisition_diversity_v3"
SOURCES = {}
COLORS = ["#72519a", "#777777", "#267ca8", "#cb8327"]


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = sha256(path)
    return json.loads(path.read_text())


def load(path):
    SOURCES[str(path.relative_to(ROOT))] = sha256(path)
    return np.load(path)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for font in Path("/tmp/trdp2-report-fonts").glob("*.TTF"):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )
    posterior = read(DATA / "posterior_comparison_v1/summary.json")
    pilot = read(DATA / "completion_matched_v2/summary.json")["report"]
    summary = read(DATA / "acquisition_diversity_v3/summary.json")
    risk = read(RUN / "risk_sensitivity.json")
    d, _ = read_design()
    SOURCES["experiments/worm_uncertainty/protocols/acquisition_diversity_v3.json"] = sha256(
        ROOT / "experiments/worm_uncertainty/protocols/acquisition_diversity_v3.json"
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.12), sharey=True, layout="constrained")
    methods = ["density_linear", "joint_linear", "joint_nonlinear"]
    for ax, key, title in zip(
        axes,
        ["voxel_coverage", "withheld_coverage"],
        ["Density coverage", "Withheld clean-projection coverage"],
    ):
        for j, (method, label, color) in enumerate(
            zip(methods, ["Density only", "Joint linear", "Joint nonlinear"], COLORS[1:])
        ):
            values = [
                100
                * next(
                    r[key]
                    for r in posterior["groups"]
                    if r["noisy"] and r["shape"] == shape and r["method"] == method
                )
                for shape in ["moderate", "elongated"]
            ]
            ax.bar(np.arange(2) + (j - 1) * 0.23, values, 0.23, label=label, color=color)
        ax.axhline(95, color="black", ls="--", lw=0.8, label="95% nominal")
        ax.set_xticks([0, 1], ["Moderate", "Elongated"])
        ax.set_ylim(0, 105)
        ax.set_title(title)
    axes[0].set_ylabel("Empirical coverage (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="outside upper center", frameon=False)
    save(fig, "phantom_coverage.pdf")
    fig, ax = plt.subplots(figsize=(6.7, 1.95), layout="constrained")
    values = {
        arm: [
            pilot["selection"]["familiar"][arm]["relative_l2"],
            pilot["shifts"][arm]["relative_l2"],
        ]
        for arm in ["interpolation", "mask", "geometry"]
    }
    for i, (arm, label, color) in enumerate(
        zip(
            values, ["Interpolation", "Mask CNN", "Geometry CNN"], [COLORS[0], COLORS[3], COLORS[2]]
        )
    ):
        bars = ax.bar(
            np.arange(2) + (i - 1) * 0.22,
            np.array(values[arm]) * 100,
            0.22,
            label=label,
            color=color,
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=1)
    ax.set_xticks([0, 1], ["Familiar validation masks", "Unseen masks, same validation files"])
    ax.set_ylabel("Missing relative L2 (%) ↓")
    ax.set_ylim(0, 65)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    save(fig, "pilot_generalization.pdf")
    a = load(RUN / "audit/inputs.npz")
    clean = load(RUN / "audit/clean.npy")
    ens = load(RUN / "audit/ensemble.npz")
    f = 0
    mi = next(
        i
        for i, r in enumerate(d["inventories"]["audit"])
        if r["group"] == "new_in_distribution" and r["count"] == 13
    )
    zi = 4
    degrees, uv, _ = grid(d)
    angles = np.deg2rad(degrees)
    u = np.unique(uv[:, 0])
    extent = [u[0] - (u[1] - u[0]) / 2, u[-1] + (u[1] - u[0]) / 2, -3.6, 176.4]
    values = [
        clean[f, zi],
        a["observed"][f, mi, zi],
        a["interpolation"][f, mi, zi],
        ens["mean"][f, mi, zi],
    ]
    spectra = [polar_spectrum(v, u, angles) for v in values]
    images = [filtered_backprojection(v, u, angles, 128) for v in values]
    fig, axes = plt.subplots(3, 4, figsize=(6.7, 4.12), layout="constrained")
    pmin = min(0, min(v.min() for v in values))
    pmax = max(np.quantile(v, 0.995) for v in values)
    smax = max(s[2].max() for s in spectra)
    imin = min(np.quantile(v, 0.005) for v in images)
    imax = max(np.quantile(v, 0.995) for v in images)
    for i, (v, s, image, label) in enumerate(
        zip(
            values,
            spectra,
            images,
            ["Clean projections", "Acquired (zero fill)", "Interpolation", "B: ensemble mean"],
        )
    ):
        im = axes[0, i].imshow(
            v, origin="lower", aspect="auto", extent=extent, vmin=pmin, vmax=pmax, cmap="viridis"
        )
        axes[0, i].set_title(label)
        axes[0, i].set_yticks([0, 90, 172.8])
        axes[0, i].set_xticks([-0.5, 0, 0.5])
        axes[0, i].set_xlabel("Detector u", labelpad=0)
        sp = axes[1, i].scatter(
            s[0].ravel(),
            s[1].ravel(),
            c=s[2].ravel(),
            s=0.45,
            vmin=0,
            vmax=smax,
            cmap="magma",
            rasterized=True,
        )
        axes[1, i].set_aspect("equal")
        axes[1, i].set_xticks([-15, 0, 15])
        axes[1, i].set_yticks([-15, 0, 15])
        axes[1, i].set_xlabel("Frequency x", labelpad=0)
        ri = axes[2, i].imshow(
            image, origin="lower", extent=[-1, 1, -1, 1], vmin=imin, vmax=imax, cmap="gray"
        )
        axes[2, i].set_xlim(-0.6, 0.6)
        axes[2, i].set_ylim(-0.6, 0.6)
        axes[2, i].set_xticks([-0.5, 0, 0.5])
        axes[2, i].set_yticks([-0.5, 0, 0.5])
        axes[2, i].set_xlabel("x (normalized)", labelpad=0)
        if i:
            for row in range(3):
                axes[row, i].set_yticklabels([])
    axes[0, 0].set_ylabel("Angle (degrees)")
    axes[1, 0].set_ylabel("Frequency y")
    axes[2, 0].set_ylabel("y (normalized)")
    for image, axs, label in [
        (im, axes[0, :], "Projection"),
        (sp, axes[1, :], "log(1 + |FT|)"),
        (ri, axes[2, :], "FBP value"),
    ]:
        fig.colorbar(image, ax=axs, shrink=0.88, pad=0.015, label=label)
    save(fig, "projection_fourier_fbp.pdf")
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.35), layout="constrained")
    for key, label, color in [
        ("disagreement", "Model disagreement", COLORS[2]),
        ("angular_distance", "Angular distance", COLORS[3]),
        ("random", "Random", COLORS[1]),
    ]:
        curves = np.array([r[key] for r in summary["risk"]])
        axes[0].plot(
            [10, 25, 50, 75, 100], curves.mean(0), marker="o", ms=3, label=label, color=color
        )
    axes[0].set_xlabel("Lowest-score missing rays retained (%)")
    axes[0].set_ylabel("Absolute clean-ray error ↓")
    axes[0].legend(frameon=False)
    axes[0].set_title("Audit error-risk ranking")
    w = risk["worst_low_disagreement"]
    fi, mi, zi, ai = w["file_index"], w["mask_index"], w["slice_index"], w["angle_index"]
    axes[1].plot(u, clean[fi, zi, ai], color="black", label="Clean target", lw=1.4)
    for i, seed in enumerate(d["training_seeds"]):
        p = load(RUN / "audit" / f"B_{seed}.npy")
        axes[1].plot(u, p[fi, mi, zi, ai], lw=0.9, alpha=0.8, label=f"Model {i + 1}")
    axes[1].axvline(u[w["detector_index"]], color="gray", ls=":", lw=0.8)
    axes[1].set_xlabel("Detector u")
    axes[1].set_ylabel("Normalized projection")
    axes[1].set_title("Agreement on a wrong prediction")
    axes[1].legend(frameon=False, ncol=2)
    save(fig, "disagreement_risk_failure.pdf")
    ref = summary["refinement"]
    case = ref["cases"][0]
    folder = RUN / "refinement" / Path(case["file"]).stem
    inputs = load(folder / "inputs.npz")
    fits = read(folder / "fits.json")
    images = [
        density_slice(
            np.asarray(
                next(
                    f["parameters"]
                    for f in fits
                    if f["arm"] == r["arm"] and f["seed"] == r["selected_seed"]
                )
            ),
            inputs["v"][4],
            160,
        )
        for r in case["rows"]
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.7, 1.75), layout="constrained")
    vmax = max(np.quantile(v, 0.995) for v in images)
    for ax, image, label in zip(
        axes,
        images,
        ["Acquired-only initializer", "Interpolation initializer", "Learned initializer"],
    ):
        im = ax.imshow(image, origin="lower", extent=[-1, 1, -1, 1], vmin=0, vmax=vmax, cmap="gray")
        ax.set_xlim(-0.6, 0.4)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im, ax=axes, pad=0.01, label="Model density")
    save(fig, "gaussian_density.pdf")
    for path in [
        "worm_staged_v1/summary.json",
        "scaled_acceptance_v1/summary.json",
        "nonlinear_posterior_v1/summary.json",
        "completion_matched_v2/sampler_summary.json",
        "acquisition_diversity_v3/descriptive_summary.json",
    ]:
        read(DATA / path)
    meta = dict(
        source_commit="c88712ca9f1dd25de0edd7512e45dab8cf66b3e1",
        sources=SOURCES,
        generator_sha256=sha256(Path(__file__)),
        figures={p.name: sha256(p) for p in OUT.iterdir()},
        note="All figures derive from saved results; no training, optimization or sampling rerun. Labels and compact layouts adapted for the report.",
    )
    (ROOT / "docs/reports/week6_sources.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("Built", len(meta["figures"]), "report figures")


if __name__ == "__main__":
    main()
