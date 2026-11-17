"""All predeclared sweep budgets and a fixed5/100-sweep density/error comparison."""

import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.worm_uncertainty.studies.sart_duration import DEFAULT


def make(root, out):
    out.mkdir(parents=True, exist_ok=False)
    for name in [
        "protocol.json",
        "provenance.json",
        "summary.json",
        "status.json",
        "metrics.csv",
        "stability.json",
        "input_hashes.json",
        "separation.json",
    ]:
        shutil.copyfile(root / name, out / name)
    cfg = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    sweeps = cfg["schedule"]["reported_sweeps"]
    labels = {
        "zero": "Zero start",
        "interpolation": "Interpolation start",
        "learned": "Learned start",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for i, span in enumerate([90, 120]):
        for j, key in enumerate(["density_l2", "withheld_l2", "acquired_l2"]):
            ax = axes[i, j]
            for arm, label in labels.items():
                ax.plot(
                    sweeps,
                    [100 * summary["curves"][str(span)][str(s)][arm][key] for s in sweeps],
                    marker="o",
                    label=label,
                )
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xticks(sweeps, [str(v) for v in sweeps])
            ax.set_xlabel("SART sweeps (fixed schedule)")
            ax.set_ylabel("Relative L2 (%) — lower is better")
            ax.set_title(
                str(span)
                + "°: "
                + {
                    "density_l2": "known density error",
                    "withheld_l2": "withheld projection error",
                    "acquired_l2": "acquired residual",
                }[key]
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Projection agreement can improve while density error increases", fontsize=12)
    fig.savefig(out / "duration_curves.png", dpi=150)
    plt.close(fig)
    parent = Path(cfg["parent"])
    truth = np.load(parent / "evaluation_arrays.npz")["truth"][1]
    saved = np.load(root / "solutions.npz")
    configs = [(arm, sweep) for sweep in [5, 100] for arm in labels]
    volumes = [truth] + [saved[f"{arm}_m2_s{sweep}"][1] for arm, sweep in configs]
    titles = ["Known truth"] + [labels[arm] + f"\n{sweep} sweeps" for arm, sweep in configs]
    hi = max(float(v.max()) for v in volumes)
    ehi = max(float(abs(v - truth).max()) for v in volumes)
    fig, axes = plt.subplots(2, len(volumes), figsize=(19, 6.5), layout="constrained")
    for j, (v, title) in enumerate(zip(volumes, titles)):
        im = axes[0, j].imshow(
            v, origin="lower", extent=(-155, 155, -155, 155), vmin=0, vmax=hi, cmap="gray"
        )
        axes[0, j].set_title(title, fontsize=10)
        er = axes[1, j].imshow(
            abs(v - truth),
            origin="lower",
            extent=(-155, 155, -155, 155),
            vmin=0,
            vmax=ehi,
            cmap="inferno",
        )
        axes[1, j].set_title("Absolute density error ↓", fontsize=9)
        for ax in axes[:, j]:
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
    fig.colorbar(im, ax=axes[0].tolist(), label="Synthetic density (model units)", shrink=0.7)
    fig.colorbar(er, ax=axes[1].tolist(), label="Absolute error; lower is better", shrink=0.7)
    fig.suptitle("Fixed elongated object1; start21.6°, acquired span90°; common display scales")
    fig.savefig(out / "duration_density.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    make(a.root, a.output)
