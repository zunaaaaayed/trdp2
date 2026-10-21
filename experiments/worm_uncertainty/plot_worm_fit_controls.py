"""Plot optimization controls with explicit error direction and acceptance status."""

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
    parser.add_argument(
        "--parent", type=Path, default=Path("output/worm_uncertainty/worm_shape_pilot_v1")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    report = json.loads((args.input / "report.json").read_text())
    parent = json.loads((args.parent / "report.json").read_text())
    if sha256(args.parent / "report.json") != report["parent_report_sha256"]:
        raise ValueError("parent report mismatch")
    for name, expected in report["artifacts_sha256"].items():
        if sha256(args.input / name) != expected:
            raise ValueError("artifact mismatch: " + name)
    if sha256(args.parent / "inputs.npz") != parent["artifacts_sha256"]["inputs.npz"]:
        raise ValueError("parent input mismatch")
    args.output.mkdir(parents=True)
    rows = [parent] + report["rows"]
    labels = [
        "Old start/prior\n16G · 150",
        "New start/old prior\n16G · 150",
        "New start/prior\n16G · 150",
        "New start/prior\n16G · 450",
        "New start/prior\n32G · 450",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(rows))
    for offset, key, label, color in [
        (-0.18, "measured_relative_l2", "Acquired", "#4477AA"),
        (0.18, "withheld_relative_l2", "Withheld", "#CC6677"),
    ]:
        bars = axes[0].bar(
            x + offset, [100 * r[key] for r in rows], width=0.35, label=label, color=color
        )
        axes[0].bar_label(bars, fmt="%.1f", fontsize=8)
    axes[0].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Clean projection relative L2 error (%) — lower is better",
        title="Error improvement does not establish calibrated uncertainty",
    )
    axes[0].tick_params(axis="x", labelsize=7)
    axes[0].set_ylim(0, 1.3 * max(100 * row["withheld_relative_l2"] for row in rows))
    axes[0].legend()
    fits = json.loads((args.input / "fits.json").read_text())
    for i, row in enumerate(rows):
        n = row["gate"]["converged_candidates"]
        passed = row["gate"]["passed"]
        axes[1].bar(i, n, color="#228833" if passed else "#AA3377")
        axes[1].text(i, n + 0.1, "PASS" if passed else "FAIL", ha="center", fontsize=9)
    axes[1].set(
        xticks=x,
        xticklabels=labels,
        ylim=(0, 4.8),
        yticks=range(5),
        ylabel="Converged starts out of 4 — more is better",
        title="Gate also requires agreement and no active bounds",
    )
    axes[1].tick_params(axis="x", labelsize=7)
    fig.suptitle("One development worm · same sparse data/noise · 120° acquisition", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output / "fit_controls.png", dpi=160)
    plt.close(fig)
    data = np.load(args.parent / "inputs.npz")
    predictions = np.load(args.input / "predictions.npz")
    shape = (len(data["angles"]), len(data["v"]), len(data["uv"]) // len(data["v"]))
    angle_index = int(np.argmin(abs(np.rad2deg(data["angles"]) - 144)))
    z_index = 4  # Same fixed source z=256 profile as the parent pilot.
    if data["measured"].reshape(shape)[angle_index].any():
        raise ValueError("profile is not withheld")
    u = data["uv"][: shape[2], 0]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        u,
        data["clean"].reshape(shape)[angle_index, z_index],
        color="black",
        lw=2,
        label="Clean withheld reference",
    )
    for i, row in enumerate(report["rows"]):
        ax.plot(
            u,
            predictions[row["arm"]["name"]].reshape(shape)[angle_index, z_index],
            label=labels[i + 1].replace("\n", " · "),
        )
    ax.set(
        xlabel="Normalized detector coordinate",
        ylabel="Normalized projection value (not accuracy)",
        title="144° withheld profile · z=256 · closer to black is better",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "withheld_profile.png", dpi=160)
    plt.close(fig)
    report["fit_diagnostics"] = [
        {k: v for k, v in fit.items() if k != "parameters"} for fit in fits
    ]
    report["parent_baseline"] = {
        k: parent[k]
        for k in ["gate", "measured_relative_l2", "withheld_relative_l2", "initializer"]
    }
    report["report_sha256"] = sha256(args.input / "report.json")
    report["plot_code_sha256"] = sha256(__file__)
    report["figure_sha256"] = {
        name: sha256(args.output / name) for name in ["fit_controls.png", "withheld_profile.png"]
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
