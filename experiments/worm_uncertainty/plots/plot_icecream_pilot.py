"""Comparable sinograms, Fourier slices, volume planes and errors; no checkpoint selection."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.worm_uncertainty.studies.icecream_pilot import DEFAULT_ROOT


def make_figures(root, output):
    output.mkdir(parents=True, exist_ok=False)
    reports = sorted((root / "evaluation").glob("*/report.json"))
    reports += sorted((root / "evaluation_fp32").glob("*/report.json"))
    rows = []
    for p in reports:
        r = json.loads(p.read_text())
        for row in r["rows"]:
            rows.append(dict(case=r["case"]["id"], **row))
    (output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    cases = [p.parent.name for p in reports]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, key, title in zip(
        axes,
        ["volume_relative_l2", "withheld_relative_l2", "acquired_noisy_relative_l2"],
        [
            "Known-volume error ↓ (phantoms only)",
            "Withheld clean-projection error ↓",
            "Acquired noisy residual ↓",
        ],
    ):
        for j, method in enumerate(["FBP", "SART", "ICECREAM"]):
            values = [
                next(v[key] for v in rows if v["case"] == c and v["method"] == method)
                for c in cases
            ]
            values = [100 * v if v is not None else np.nan for v in values]
            ax.bar(np.arange(len(cases)) + (j - 1) * 0.25, values, width=0.25, label=method)
        ax.set_xticks(
            range(len(cases)),
            [c.replace("phantom_", "p_").replace("rotated", "rot") for c in cases],
            rotation=65,
            ha="right",
            fontsize=7,
        )
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Relative L2 (%) — lower is better")
        ax.grid(axis="y", alpha=0.2)
    axes[-1].legend(fontsize=8)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    for case_id in ["phantom_base_90", "worm004_120", "worm012_90"]:
        if case_id not in cases:
            continue
        case = root / "inputs" / case_id
        ev = next(p.parent for p in reports if p.parent.name == case_id)
        truth_path = case / "evaluation_only/truth.npy"
        volumes = [np.load(ev / (m + "_volume.npy")) for m in ["fbp", "sart", "icecream"]]
        titles = ["FBP pair average", "Nonnegative SART", "ICECREAM (per scan)"]
        if truth_path.exists():
            ref = np.load(truth_path)
            volumes.insert(0, ref)
            titles.insert(0, "Known phantom truth")
        else:
            # No full-angle reconstruction enters training or serves as voxel truth.
            ref = None
        ncol = len(volumes)
        nz, ny, nx = volumes[0].shape
        lo = min(0, min(float(v.min()) for v in volumes))
        hi = max(float(v.max()) for v in volumes)
        fig, axes = plt.subplots(4, ncol, figsize=(3.2 * ncol, 10), layout="constrained")
        spectra = [np.log1p(np.abs(np.fft.fftshift(np.fft.fftn(v)))) for v in volumes]
        shi = max(float(s.max()) for s in spectra)
        for j, (v, title, spec) in enumerate(zip(volumes, titles, spectra)):
            for i, plane in enumerate([v[nz // 2], v[:, ny // 2], v[:, :, nx // 2]]):
                im = axes[i, j].imshow(
                    plane, cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect="equal"
                )
                axes[i, j].set_title(
                    title if i == 0 else ["", "XZ central plane", "YZ central plane"][i], fontsize=9
                )
            fm = axes[3, j].imshow(spec[nz // 2], cmap="magma", vmin=0, vmax=shi, origin="lower")
            axes[3, j].set_title("Fourier XY: log(1+|FFT|)", fontsize=9)
        fig.colorbar(
            im,
            ax=axes[:3].ravel().tolist(),
            shrink=0.6,
            label="Model intensity (not an accuracy score)",
        )
        fig.colorbar(
            fm,
            ax=axes[3].ravel().tolist(),
            shrink=0.65,
            label="Fourier magnitude (not an accuracy score)",
        )
        fig.suptitle(
            case_id
            + (
                ": known truth"
                if ref is not None
                else ": no voxel ground truth; morphology remains unverified"
            ),
            fontsize=11,
        )
        fig.savefig(output / (case_id + "_volume_fourier.png"), dpi=150)
        plt.close(fig)
        clean = np.load(case / "evaluation_only/clean.npy")[nz // 2]
        detector_half_width = clean.shape[-1] / 2
        predictions = [
            np.load(ev / (m + "_projection.npy"))[nz // 2] for m in ["fbp", "sart", "icecream"]
        ]
        info = json.loads((case / "report.json").read_text())
        mask = np.zeros(100, bool)
        mask[info["mask"]["indices"]] = True
        lim = max(float(abs(v - clean).max()) for v in predictions)
        fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
        for j, (v, title) in enumerate(
            zip(
                [clean] + predictions,
                [
                    "Clean target (evaluation only)",
                    "FBP reprojection",
                    "SART reprojection",
                    "ICECREAM reprojection",
                ],
            )
        ):
            im = axes[0, j].imshow(
                v,
                aspect="auto",
                vmin=float(clean.min()),
                vmax=float(clean.max()),
                cmap="gray",
                extent=(-detector_half_width, detector_half_width, 179.1, -0.9),
            )
            axes[0, j].set_title(title, fontsize=9)
            if j:
                err = axes[1, j].imshow(
                    abs(v - clean),
                    aspect="auto",
                    vmin=0,
                    vmax=lim,
                    cmap="inferno",
                    extent=(-detector_half_width, detector_half_width, 179.1, -0.9),
                )
                axes[1, j].set_title("Absolute error ↓", fontsize=9)
            else:
                axes[1, j].imshow(
                    np.broadcast_to(mask[:, None], clean.shape),
                    aspect="auto",
                    vmin=0,
                    vmax=1,
                    cmap="gray",
                    extent=(-detector_half_width, detector_half_width, 179.1, -0.9),
                )
                axes[1, j].set_title("White = measured; black = withheld", fontsize=9)
            axes[1, j].set_xlabel("Detector position (µm)")
            for ax in axes[:, j]:
                ax.set_ylabel("Angle (degrees)")
        fig.colorbar(im, ax=axes[0].tolist(), label="Fixed normalized projection intensity")
        fig.colorbar(err, ax=axes[1, 1:].tolist(), label="Smaller error is better")
        fig.savefig(output / (case_id + "_sinograms.png"), dpi=150)
        plt.close(fig)
        if ref is not None:
            errors = [abs(v - ref) for v in volumes[1:]]
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), layout="constrained")
            vmax = max(float(e.max()) for e in errors)
            for ax, error, title in zip(axes, errors, titles[1:]):
                im = ax.imshow(error[nz // 2], vmin=0, vmax=vmax, cmap="inferno", origin="lower")
                ax.set_title(title)
            fig.colorbar(im, ax=axes.tolist(), label="Absolute voxel error ↓; lower is better")
            fig.savefig(output / (case_id + "_voxel_errors.png"), dpi=150)
            plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for case_id in cases:
        group = "runs_fp32" if (root / "runs_fp32" / case_id).exists() else "runs"
        curve = np.genfromtxt(root / group / case_id / "losses.csv", delimiter=",", names=True)
        width = 100
        value = curve["total_loss"]
        ax.plot(
            curve["update"][width - 1 :],
            np.convolve(value, np.ones(width) / width, "valid"),
            label=case_id,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Optimizer updates")
    ax.set_ylabel("Self-supervised loss (100-update mean) ↓")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Lower training loss does not guarantee lower withheld error")
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    make_figures(a.root, a.output)
