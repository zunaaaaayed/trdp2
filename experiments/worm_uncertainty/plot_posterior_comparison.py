"""Show coverage, widths, error and a predeclared slice without hiding exclusions."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from prepare import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    report = json.loads((args.input / "report.json").read_text())
    for n, h in report["artifacts_sha256"].items():
        if sha256(args.input / n) != h:
            raise ValueError("artifact hash mismatch: " + n)
    if len(report["rows"]) != 31 or len(report["excluded"]) != 1:
        raise ValueError("cohort mismatch")
    groups = []
    for noisy in [False, True]:
        for shape in ["moderate", "elongated"]:
            cases = [
                r
                for r in report["rows"]
                if (r["noise_seed"] is not None) == noisy and r["shape"] == shape
            ]
            for method in report["protocol"]["methods"]:
                vs = [r["methods"][method]["voxel"]["foreground"] for r in cases]
                ps = [r["methods"][method]["projection"]["withheld_foreground"] for r in cases]
                groups.append(
                    dict(
                        noisy=noisy,
                        shape=shape,
                        method=method,
                        cases=len(cases),
                        voxel_coverage=float(np.mean([v["coverage95"] for v in vs])),
                        voxel_coverage_min=float(min(v["coverage95"] for v in vs)),
                        voxel_coverage_max=float(max(v["coverage95"] for v in vs)),
                        voxel_width_relative_peak=float(
                            np.mean([v["width_relative_peak"] for v in vs])
                        ),
                        voxel_mean_relative_error=float(np.mean([v["relative_l2"] for v in vs])),
                        withheld_coverage=float(np.mean([v["coverage95"] for v in ps])),
                        withheld_width_relative_peak=float(
                            np.mean([v["width_relative_peak"] for v in ps])
                        ),
                        max_split_half_coverage_difference=float(
                            max(v["half_sample_coverage_difference"] for v in vs)
                        ),
                        max_split_half_width_difference=float(
                            max(v["half_sample_width_relative_difference"] for v in vs)
                        ),
                    )
                )
    args.output.mkdir(parents=True)
    methods = ["density_linear", "joint_linear", "joint_nonlinear"]
    labels = ["Density only", "Joint / linear", "Joint / nonlinear"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for i, shape in enumerate(["moderate", "elongated"]):
        gs = [
            next(g for g in groups if g["noisy"] and g["shape"] == shape and g["method"] == m)
            for m in methods
        ]
        for j, (key, title, ylabel) in enumerate(
            [
                (
                    "voxel_coverage",
                    "Coverage: target 95%",
                    "Foreground truth covered (%)",
                ),
                (
                    "voxel_width_relative_peak",
                    "Width: assess alongside coverage",
                    "Mean interval width / queried truth peak (%)",
                ),
                (
                    "voxel_mean_relative_error",
                    "Mean error: lower is better",
                    "Foreground relative L2 error (%)",
                ),
            ]
        ):
            bars = axes[i, j].bar(
                range(3), [100 * g[key] for g in gs], color=["#4477AA", "#228833", "#CC6677"]
            )
            axes[i, j].bar_label(bars, fmt="%.1f", fontsize=8)
            axes[i, j].set(
                xticks=range(3),
                xticklabels=labels,
                ylabel=ylabel,
                title=f"{shape.capitalize()} (n={gs[0]['cases']})\n{title}",
            )
            axes[i, j].tick_params(axis="x", rotation=20)
            if j == 0:
                axes[i, j].axhline(95, color="black", ls="--")
                axes[i, j].set_ylim(0, 108)
            else:
                axes[i, j].set_ylim(0, max(1e-6, max(100 * g[key] for g in gs)) * 1.2)
    fig.suptitle(
        "Noisy eligible cases only (23/24) · oracle geometry prior · local Gaussian diagnostic"
    )
    fig.tight_layout()
    fig.savefig(args.output / "coverage_width_error.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, shape in zip(axes, ["moderate", "elongated"]):
        cases = [r for r in report["rows"] if r["shape"] == shape and r["noise_seed"] is not None]
        for shift, region, label in [
            (-0.18, "acquired_foreground", "Acquired"),
            (0.18, "withheld_foreground", "Withheld"),
        ]:
            values = [
                100
                * np.mean([r["methods"][method]["projection"][region]["coverage95"] for r in cases])
                for method in methods
            ]
            bars = ax.bar(np.arange(3) + shift, values, width=0.35, label=label)
            ax.bar_label(bars, fmt="%.1f", fontsize=8)
        ax.axhline(95, color="black", ls="--")
        ax.set(
            xticks=range(3),
            xticklabels=labels,
            ylim=(0, 108),
            ylabel="Latent clean projection coverage (%)",
            title=f"{shape.capitalize()}: target 95%, assess width too",
        )
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Projection consistency: noisy eligible cases · not observation-predictive intervals"
    )
    fig.tight_layout()
    fig.savefig(args.output / "projection_coverage.png", dpi=160)
    plt.close(fig)
    visual = np.load(args.input / "visual.npz")
    axis = np.load(args.input / "queries.npz")["slice_axis"]
    truth = visual["truth"].reshape(len(axis), len(axis))
    limit = float(truth.max())
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    arrays = [
        truth,
        visual["map"].reshape(truth.shape),
        visual["joint_linear_mean"].reshape(truth.shape),
        visual["joint_nonlinear_mean"].reshape(truth.shape),
    ]
    titles = ["Known truth", "Selected fit", "Joint linear mean", "Joint nonlinear mean"]
    for ax, array, title in zip(axes[0], arrays, titles):
        im = ax.imshow(
            array.T, origin="lower", extent=[-1, 1, -1, 1], vmin=0, vmax=limit, cmap="gray"
        )
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.7)
    errors = [abs(a - truth) for a in arrays[1:]]
    error_limit = max(float(a.max()) for a in errors)
    for ax, array, title in zip(
        axes[1, :3],
        errors,
        ["Fit absolute error", "Linear mean absolute error", "Nonlinear mean absolute error"],
    ):
        im = ax.imshow(
            array.T, origin="lower", extent=[-1, 1, -1, 1], vmin=0, vmax=error_limit, cmap="magma"
        )
        ax.set_title(title + "\nLower is better")
        fig.colorbar(im, ax=ax, shrink=0.7)
    im = axes[1, 3].imshow(
        visual["joint_nonlinear_sd"].reshape(truth.shape).T,
        origin="lower",
        extent=[-1, 1, -1, 1],
        cmap="viridis",
        vmin=0,
    )
    axes[1, 3].set_title("Nonlinear sample SD\nLow SD does not imply accuracy")
    fig.colorbar(im, ax=axes[1, 3], shrink=0.7)
    fig.suptitle(
        report["protocol"]["designated_visual_case"] + " · fixed z=0 slice · model density units"
    )
    fig.tight_layout()
    fig.savefig(args.output / "designated_slice.png", dpi=160)
    plt.close(fig)
    summary = dict(
        groups=groups,
        excluded=report["excluded"],
        eligible_clean=report["eligible_clean"],
        eligible_noisy=report["eligible_noisy"],
        cases=report["rows"],
        elapsed_seconds=report["elapsed_seconds"],
        protocol=report["protocol"],
        report_sha256=sha256(args.input / "report.json"),
        plot_code_sha256=sha256(__file__),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
