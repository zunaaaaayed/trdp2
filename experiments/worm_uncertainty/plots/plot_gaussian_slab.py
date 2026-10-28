"""Readable figure from saved pilot arrays; no reconstruction rerun."""

import argparse
import json
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.studies.check_fbp import plt
from experiments.worm_uncertainty.data.prepare import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("output exists")
    report = json.loads((a.input / "report.json").read_text())
    if sha256(a.input / "arrays.npz") != report["artifacts"]["arrays.npz"]:
        raise ValueError("input hash mismatch")
    data = np.load(a.input / "arrays.npz")
    ref = data["initial_fbp_zyx"][4]
    vol = data["volume_xyz"][:, :, 4].T
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    lo = min(ref.min(), vol.min())
    hi = max(ref.max(), vol.max())
    for ax, array, title in [
        (axes[0], ref, "FBP initialization"),
        (axes[1], vol, "3D Gaussians: 300 updates"),
    ]:
        im = ax.imshow(
            array[105:205, 105:205], extent=[105, 205, 205, 105], vmin=lo, vmax=hi, cmap="gray"
        )
        ax.set_title(title)
        ax.set_xlabel("x pixel")
        ax.set_ylabel("y pixel")
    fig.colorbar(im, ax=axes[:2].tolist(), shrink=0.65, label="Uncalibrated density units")
    axes[2].plot(data["loss"])
    axes[2].set(
        xlabel="update", ylabel="measured-view MSE", title="Lower loss = better data agreement"
    )
    fig.suptitle(
        "Brightness measures density, not accuracy. FBP is not ground truth.\nSame crop and scale; data fit alone does not establish voxel correctness.",
        fontsize=11,
    )
    fig.savefig(a.output, dpi=140)
    plt.close(fig)
    a.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                input_arrays_sha256=sha256(a.input / "arrays.npz"),
                renderer_sha256=sha256(__file__),
                figure_sha256=sha256(a.output),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
