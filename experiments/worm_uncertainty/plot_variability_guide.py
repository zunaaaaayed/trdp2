"""Render saved phantom results with interpretation labels; no reconstruction."""

import argparse
import json
from pathlib import Path
import numpy as np
from check_fbp import plt
from prepare import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    report = json.loads((args.input / "report.json").read_text())
    if sha256(args.input / "arrays.npz") != report["artifacts"]["arrays.npz"]:
        raise ValueError("input hash mismatch")
    a = np.load(args.input / "arrays.npz")
    truth = a["truth"]
    spans = [180, 120, 90]
    vmax = max(
        max(
            abs(a[f"reconstructions_{s}"].mean(0) - truth).max(),
            a[f"reconstructions_{s}"].std(0, ddof=1).max(),
        )
        for s in spans
    )
    fig, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
    for i, span in enumerate(spans):
        stack = a[f"reconstructions_{span}"]
        mean = stack.mean(0)
        panels = [
            (truth, "Known density\nbrightness is not quality"),
            (mean, "Mean reconstruction\ncloser to truth is better"),
            (abs(mean - truth), "Absolute bias\n0 is ideal; higher is worse"),
            (stack.std(0, ddof=1), "Noise variability\nlower = more repeatable, not more accurate"),
        ]
        for j, (v, title) in enumerate(panels):
            im = axes[i, j].imshow(
                v, vmin=0, vmax=1.2 if j < 2 else vmax, cmap="gray" if j < 2 else "magma"
            )
            axes[i, j].set_title(f"{span}°: {title}", fontsize=8)
            axes[i, j].set_axis_off()
            if i == 2:
                fig.colorbar(
                    im, ax=axes[:, j].tolist(), shrink=0.55, label="arbitrary density units"
                )
    fig.suptitle(
        "How to read this figure: a dark variability map can still accompany large systematic errors.\nError and variability use the same scale; compare their numeric magnitudes.",
        fontsize=11,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                input_arrays_sha256=sha256(args.input / "arrays.npz"),
                renderer_sha256=sha256(__file__),
                figure_sha256=sha256(args.output),
                error_and_variability_max=float(vmax),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
