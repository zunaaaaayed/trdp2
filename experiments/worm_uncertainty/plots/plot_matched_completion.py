"""Matched completion validation: paired files, acquisition shifts and fixed slice."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.worm_uncertainty.data.matched_completion import evaluation_masks
from experiments.worm_uncertainty.data.prepare import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("output/worm_uncertainty/completion_matched_v2")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    args.output.mkdir(parents=True)
    report = json.loads((args.source / "report.json").read_text())
    design = json.loads((args.source / "protocol.json").read_text())
    arms = ["interpolation", "mask", "geometry"]
    labels = ["Interpolation", "Values + mask", "Values + mask + geometry"]
    colors = ["#999999", "#4c72b0", "#dd8452"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for i, (arm, label, color) in enumerate(zip(arms, labels, colors)):
        rows = report["selection"]["familiar"][arm]["per_file"]
        axes[0].bar(
            np.arange(4) + (i - 1) * 0.25,
            [100 * r["relative_l2"] for r in rows],
            0.25,
            label=label,
            color=color,
        )
        rows = report["shifts"][arm]["per_mask_file"]
        configs = design["shift_masks"]
        axes[1].bar(
            np.arange(len(configs)) + (i - 1) * 0.25,
            [
                100
                * np.mean([r["relative_l2"] for r in rows if r["start"] == a and r["span"] == s])
                for a, s in configs
            ],
            0.25,
            color=color,
        )
    axes[0].set_xticks(np.arange(4), ["153", "204", "3579", "4047"])
    axes[0].set_xlabel("Validation deformation ID (one organism)")
    axes[0].set_title("Familiar masks: learned completion improves all four files")
    axes[1].set_xticks(np.arange(5), [f"{a}° / {s}°" for a, s in configs])
    axes[1].set_xlabel("Unseen configuration: start / span")
    axes[1].set_title("Unseen 90° spans expose transfer failure")
    for ax in axes:
        ax.set_ylabel("Missing-projection relative L2 (%) — lower is better")
        ax.set_ylim(0, 90)
    axes[0].legend(fontsize=8)
    fig.savefig(args.output / "validation_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for arm, label in zip(arms[1:], labels[1:]):
        history = json.loads((args.source / (arm + "_history.json")).read_text())
        ax.plot(
            [r["epoch"] for r in history],
            [100 * r["validation"]["relative_l2"] for r in history],
            label=label,
        )
    ax.axhline(
        100 * report["selection"]["familiar"]["interpolation"]["relative_l2"],
        ls="--",
        color="black",
        label="Interpolation",
    )
    ax.set(
        xlabel="Epoch",
        ylabel="Familiar validation error (%) — lower is better",
        title="Checkpoint selection uses familiar validation only",
    )
    ax.legend()
    fig.savefig(args.output / "training.png", dpi=180)
    plt.close(fig)
    arrays = np.load(args.source / "familiar.npz")
    ids = arrays["identities"]
    index = int(np.flatnonzero((ids[:, 0] == 0) & (ids[:, 1] == 0) & (ids[:, 2] == 4))[0])
    mask = evaluation_masks(arrays, design, design["familiar_masks"])[index]
    truth = arrays["target"][index]
    limit = float(truth.max())
    preds = [arrays[arm][index] for arm in arms]
    errors = [abs(p - truth) for p in preds]
    emax = max(float(e.max()) for e in errors)
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), layout="constrained")
    titles = [
        "Measured input (zeros accompanied by mask)",
        "Clean target (evaluation only)",
        "Acquired rows: white; missing: black",
    ]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=9)
    image = axes[0, 0].imshow(
        arrays["measured"][index], vmin=0, vmax=limit, aspect="auto", origin="lower"
    )
    axes[0, 1].imshow(truth, vmin=0, vmax=limit, aspect="auto", origin="lower")
    axes[0, 2].imshow(
        np.broadcast_to(mask[:, None], truth.shape),
        vmin=0,
        vmax=1,
        cmap="gray",
        aspect="auto",
        origin="lower",
    )
    for i, (pred, label, error) in enumerate(zip(preds, labels, errors)):
        axes[1, i].imshow(pred, vmin=0, vmax=limit, aspect="auto", origin="lower")
        axes[1, i].set_title(label, fontsize=9)
        err = axes[2, i].imshow(
            error, vmin=0, vmax=emax, cmap="magma", aspect="auto", origin="lower"
        )
        axes[2, i].set_title("Absolute error — darker is better", fontsize=9)
    for ax in axes.ravel():
        ax.set_xlabel("Sparse detector index")
        ax.set_ylabel("Sparse angle row (7.2° spacing)")
    fig.colorbar(
        image,
        ax=list(axes[0, :2]) + list(axes[1]),
        label="Normalized projection value; shared scale (display clipping only)",
        shrink=0.7,
    )
    fig.colorbar(
        err, ax=list(axes[2]), label="Absolute error; shared scale, lower is better", shrink=0.8
    )
    fig.suptitle(
        "Predeclared example: worm153, z256, start0°, span120° — not voxel truth", fontsize=11
    )
    fig.savefig(args.output / "fixed_example.png", dpi=180)
    plt.close(fig)
    summary = dict(
        report=report,
        protocol=design,
        source_report_sha256=sha256(args.source / "report.json"),
        plot_code_sha256=sha256(Path(__file__)),
        example=dict(
            file="worm_153_gt.tiff",
            z=256,
            start=0,
            span=120,
            projection_display_limits=[0, limit],
            absolute_error_limits=[0, emax],
        ),
        artifact_note="Original mask key collided with mask-arm predictions; acquisition masks recovered from frozen protocol. Metrics verified unchanged; repair record retained.",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
