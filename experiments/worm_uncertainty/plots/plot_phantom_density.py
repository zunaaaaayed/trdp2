"""Fixed known-truth panels, learning curves and grouped density comparisons."""

import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.worm_uncertainty.studies.phantom_density import DEFAULT
from experiments.worm_uncertainty.core.voxel_parallel import ParallelProjector


def make(root, out):
    out.mkdir(parents=True, exist_ok=False)
    for name in [
        "protocol.json",
        "provenance.json",
        "calibration.json",
        "masks.json",
        "training.json",
        "history.json",
        "input_hashes.json",
        "advancement_gate.json",
        "refinement.json",
        "metrics.csv",
        "summary.json",
        "status.json",
    ]:
        shutil.copyfile(root / name, out / name)
    rows = list(csv.DictReader((root / "metrics.csv").open()))
    summary = json.loads((root / "summary.json").read_text())
    methods = list(summary["primary"])
    labels = {
        "acquired_fbp": "Acquired FBP",
        "sart_zero": "SART, zero start",
        "interpolation_fbp": "Interpolation + FBP",
        "learned_fbp": "Learned + FBP",
        "sart_interpolation_init": "SART, interpolation start",
        "sart_learned_init": "SART, learned start",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    for ax, key, factor, title in zip(
        axes,
        ["density_l2", "withheld_l2", "boundary_um"],
        [100, 100, 1],
        ["Density error (%) ↓", "Withheld projection error (%) ↓", "Boundary distance (µm) ↓"],
    ):
        for j, (section, span) in enumerate([("primary", 90), ("moderate", 120)]):
            ax.bar(
                np.arange(len(methods)) + (j - 0.5) * 0.36,
                [summary[section][m][key] * factor for m in methods],
                width=0.36,
                label=str(span) + "° acquired",
            )
        ax.set_xticks(
            range(len(methods)), [labels[m] for m in methods], rotation=50, ha="right", fontsize=8
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)
    arr = np.load(root / "evaluation_arrays.npz")
    masks = json.loads((root / "masks.json").read_text())["evaluation"]
    mi = 2
    oi = 1
    truth = arr["truth"][oi]
    selected = ["sart_zero", "interpolation_fbp", "learned_fbp"] + [
        m for m in ["sart_interpolation_init", "sart_learned_init"] if m in methods
    ]
    volumes = [truth] + [arr[m][mi, oi] for m in selected]
    titles = ["Known density truth"] + [labels[m] for m in selected]
    lo = min(float(v.min()) for v in volumes)
    hi = max(float(v.max()) for v in volumes)
    ehi = max(float(abs(v - truth).max()) for v in volumes)
    specs = [np.log1p(abs(np.fft.fftshift(np.fft.fft2(v)))) for v in volumes]
    shi = max(float(v.max()) for v in specs)
    fig, axes = plt.subplots(3, len(volumes), figsize=(3 * len(volumes), 9), layout="constrained")
    for j, (v, title, spec) in enumerate(zip(volumes, titles, specs)):
        im = axes[0, j].imshow(
            v, origin="lower", extent=(-155, 155, -155, 155), vmin=lo, vmax=hi, cmap="gray"
        )
        axes[0, j].set_title(title, fontsize=9)
        err = axes[1, j].imshow(
            abs(v - truth),
            origin="lower",
            extent=(-155, 155, -155, 155),
            vmin=0,
            vmax=ehi,
            cmap="inferno",
        )
        axes[1, j].set_title("Absolute density error ↓", fontsize=9)
        fft = axes[2, j].imshow(
            spec, origin="lower", extent=(-0.1, 0.1, -0.1, 0.1), vmin=0, vmax=shi, cmap="magma"
        )
        axes[2, j].set_title("log(1+|2D FFT|)", fontsize=9)
        axes[2, j].set_xlabel("Spatial frequency (cycles/µm)")
        for ax in axes[:2, j]:
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
    fig.colorbar(
        im,
        ax=axes[0].tolist(),
        label="Synthetic density scale; brightness is not correctness",
        shrink=0.8,
    )
    fig.colorbar(err, ax=axes[1].tolist(), label="Absolute error; lower is better", shrink=0.8)
    fig.colorbar(
        fft, ax=axes[2].tolist(), label="Fourier magnitude; not an accuracy score", shrink=0.8
    )
    fig.suptitle(
        "Fixed example: elongated evaluation object 1; start21.6°, acquired span90°", fontsize=12
    )
    fig.savefig(out / "known_density_planes.png", dpi=140)
    plt.close(fig)
    op = ParallelProjector(62, np.arange(25) * 7.2, spacing=5)
    clean = arr["clean"][oi]
    mask = np.array(masks[mi]["mask"])
    projections = [clean] + [op.forward(v[None])[0] for v in volumes[1:]]
    eh = max(float(abs(p - clean).max()) for p in projections)
    fig, axes = plt.subplots(
        2, len(projections), figsize=(3 * len(projections), 6), layout="constrained"
    )
    for j, (p, title) in enumerate(zip(projections, titles)):
        im = axes[0, j].imshow(
            p,
            aspect="auto",
            extent=(-155, 155, 176.4, -3.6),
            vmin=float(clean.min()),
            vmax=float(clean.max()),
            cmap="gray",
        )
        axes[0, j].set_title(title, fontsize=9)
        if j == 0:
            axes[1, j].imshow(
                np.broadcast_to(mask[:, None], clean.shape),
                aspect="auto",
                extent=(-155, 155, 176.4, -3.6),
                vmin=0,
                vmax=1,
                cmap="gray",
            )
            axes[1, j].set_title("White=measured; black=withheld", fontsize=8)
        else:
            er = axes[1, j].imshow(
                abs(p - clean),
                aspect="auto",
                extent=(-155, 155, 176.4, -3.6),
                vmin=0,
                vmax=eh,
                cmap="inferno",
            )
            axes[1, j].set_title("Reprojection absolute error ↓", fontsize=8)
        for ax in axes[:, j]:
            ax.set_xlabel("Detector (µm)")
            ax.set_ylabel("Angle (degrees)")
    fig.colorbar(im, ax=axes[0].tolist(), label="Projection intensity")
    fig.colorbar(er, ax=axes[1, 1:].tolist(), label="Absolute error; lower is better")
    fig.savefig(out / "known_density_sinograms.png", dpi=140)
    plt.close(fig)
    history = json.loads((root / "history.json").read_text())
    fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
    ax.plot([r["update"] for r in history], [100 * r["selection"] for r in history], marker="o")
    ax.set(
        xlabel="Updates",
        ylabel="Selection missing-ray error (%) ↓",
        title="Selection performance; final update selected",
    )
    fig.savefig(out / "learning.png", dpi=150)
    plt.close(fig)
    grouped = []
    for group in ["family", "orientation"]:
        for value in sorted({r[group] for r in rows}):
            for method in methods:
                rr = [
                    r
                    for r in rows
                    if r[group] == value and r["method"] == method and float(r["span"]) == 90
                ]
                means = [
                    np.mean([float(r["density_l2"]) for r in rr if r["object"] == obj])
                    for obj in sorted({r["object"] for r in rr})
                ]
                grouped.append(
                    dict(
                        group=group,
                        value=value,
                        method=method,
                        density_l2=float(np.mean(means)),
                        objects=len(means),
                    )
                )
    (out / "grouped.json").write_text(json.dumps(grouped, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    make(a.root, a.output)
