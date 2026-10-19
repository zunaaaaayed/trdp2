"""Plot the small worm shape diagnostic with explicit stability and scale labels."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prepare import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.input / "report.json").read_text())
    cfg = report["protocol"]
    for filename, digest in report["artifacts_sha256"].items():
        if sha256(args.input / filename) != digest:
            raise ValueError("artifact checksum mismatch: " + filename)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = np.load(args.input / "inputs.npz")
    arrays = np.load(args.input / "arrays.npz")
    status = "passed" if report["gate"]["passed"] else "FAILED — provisional diagnostics"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    methods = ["density", "full_shape"]
    labels = ["Density only", "Full shape"]
    for i, (domain, label) in enumerate(
        [("withheld_latent", "Latent vs clean"), ("withheld_predictive", "Predictive vs noisy")]
    ):
        x = np.arange(2) + (i - 0.5) * 0.3
        axes[0].bar(
            x, [report["metrics"][m][domain]["coverage95"] * 100 for m in methods], 0.3, label=label
        )
        axes[1].bar(
            x, [report["metrics"][m][domain]["mean_width95"] for m in methods], 0.3, label=label
        )
    axes[0].axhline(95, ls="--", color="black")
    axes[0].set(
        title="Coverage: target ≈95%", ylabel="Withheld foreground ray coverage (%)", ylim=(0, 105)
    )
    axes[0].legend(fontsize=8)
    axes[1].set(
        title="Interval width: useful only with adequate coverage",
        ylabel="Normalized projection units",
    )
    for ax in axes:
        ax.set_xticks(np.arange(2), labels)
    fig.suptitle(
        f"Worm pilot: same fitted mean; different covariance blocks\nOptimizer stability check: {status}"
    )
    fig.savefig(args.output / "coverage.png", dpi=150)
    plt.close(fig)
    shape = (len(cfg["angle_indices"]), len(cfg["z_indices"]), len(cfg["detector_indices"]))
    index = cfg["angle_indices"].index(80)
    z = cfg["z_indices"].index(256)
    clean = inputs["clean"].reshape(shape)[index, z]
    noisy = inputs["noisy"].reshape(shape)[index, z]
    mean = arrays["prediction"].reshape(shape)[index, z]
    sd = arrays["full_sd"].reshape(shape)[index, z]
    predictive = np.sqrt(sd**2 + cfg["noise_sigma"] ** 2)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    x = np.asarray(cfg["detector_indices"])
    axes[0].plot(x, clean, label="Clean withheld reference")
    axes[0].scatter(x, noisy, s=8, label="Noisy withheld data")
    axes[0].plot(x, mean, label="Gaussian prediction")
    axes[0].fill_between(
        x,
        mean - 1.96 * predictive,
        mean + 1.96 * predictive,
        alpha=0.2,
        label="95% predictive interval",
    )
    axes[0].set(
        title="144° withheld view, source z=256",
        xlabel="Source detector index",
        ylabel="Normalized projection units",
    )
    axes[0].legend(fontsize=7)
    for ax, data, title in zip(
        axes[1:],
        [arrays["volume"][:, :, z].T, arrays["volume_sd"][:, :, z].T],
        ["Reconstructed density: brightness ≠ accuracy", "Density SD: low alone ≠ accuracy"],
    ):
        im = ax.imshow(
            data,
            origin="lower",
            extent=(-1, 1, -1, 1),
            vmin=0,
            cmap="viridis" if ax is axes[1] else "magma",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (normalized)")
        fig.colorbar(
            im, ax=ax, orientation="horizontal", label="Relative density units; uncalibrated"
        )
    fig.suptitle(
        f"Small 16-Gaussian pilot; no voxel ground truth\nOptimizer stability check: {status}"
    )
    fig.savefig(args.output / "profile_and_maps.png", dpi=150)
    plt.close(fig)
    (args.output / "results.json").write_text(
        json.dumps(
            dict(
                report=report,
                fits=json.loads((args.input / "fits.json").read_text()),
                source_report_sha256=sha256(args.input / "report.json"),
                plot_code_sha256=sha256(__file__),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
