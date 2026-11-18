"""CPU development diagnostic; scikit-image FBP is not the Gaussian backend."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skimage
import tifffile
from skimage.transform import radon, iradon
from experiments.worm_uncertainty.data.prepare import sha256


def reconstruct(sino, angles):
    # Stored input is angle x detector; skimage expects detector x angle.
    return iradon(
        sino.T,
        theta=angles,
        output_size=sino.shape[1],
        filter_name="ramp",
        circle=True,
        preserve_range=True,
    )


def phantom_check():
    n = 128
    yy, xx = np.mgrid[:n, :n]
    radius, dx, dy = 20, 11, -7
    phantom = (((xx - n // 2 - dx) ** 2 + (yy - n // 2 - dy) ** 2) <= radius**2).astype(float)
    angles = np.arange(100) * 1.8
    # Exact continuous disk chord lengths, sampled at detector centres.
    detector = np.arange(n) - n // 2
    centres = dx * np.cos(np.deg2rad(angles)) - dy * np.sin(np.deg2rad(angles))
    distance = detector[:, None] - centres[None, :]
    analytic = 2 * np.sqrt(np.maximum(radius**2 - distance**2, 0))
    projected = radon(phantom, theta=angles, circle=True, preserve_range=True)
    relative = float(np.linalg.norm(projected - analytic) / np.linalg.norm(analytic))
    recovered = reconstruct(analytic.T, angles)
    # Threshold at half the known disk density to exclude FBP ringing.
    weights = (recovered > 0.5).astype(float)
    centroid = [
        float((weights * xx).sum() / weights.sum()),
        float((weights * yy).sum() / weights.sum()),
    ]
    centre_error = float(np.linalg.norm(np.array(centroid) - [n // 2 + dx, n // 2 + dy]))
    interior = ((xx - n // 2 - dx) ** 2 + (yy - n // 2 - dy) ** 2) < (radius - 4) ** 2
    density = float(recovered[interior].mean())
    linear_error = float(np.max(np.abs(reconstruct(analytic.T * 3, angles) - recovered * 3)))
    checks = dict(
        analytic_projection_relative_l2=relative,
        centroid_error_pixels=centre_error,
        interior_mean_expected_one=density,
        linearity_max_error=linear_error,
    )
    passed = (
        relative < 0.06 and centre_error < 0.5 and abs(density - 1) < 0.05 and linear_error < 1e-10
    )
    checks["passed"] = bool(passed)
    checks["thresholds"] = dict(
        projection_relative_l2=0.06,
        centroid_pixels=0.5,
        interior_mean_absolute_error=0.05,
        linearity_max_error=1e-10,
    )
    if not passed:
        raise ValueError("phantom validation failed: " + json.dumps(checks))
    return checks, phantom, recovered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--z", type=int, default=255)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists; use a fresh destination")
    checks, phantom, recovered = phantom_check()
    if not 0 <= args.z < 510:
        raise ValueError("invalid axial slice")
    with tifffile.TiffFile(args.source) as tif:
        if tif.series[0].shape != (510, 100, 310):
            raise ValueError("unexpected TIFF shape")
        sino = tif.pages[args.z].asarray().astype(float)
    if not np.isfinite(sino).all():
        raise ValueError("nonfinite source")
    angles = np.arange(100) * 1.8
    volume = reconstruct(sino, angles)
    predicted = radon(volume, theta=angles, circle=True, preserve_range=True).T
    residual = predicted - sino
    report = dict(
        kind="cpu_full_angle_fbp_development_diagnostic",
        source=str(args.source.resolve()),
        source_sha256=sha256(args.source),
        z=args.z,
        phantom=checks,
        skimage_version=skimage.__version__,
        script_sha256=sha256(__file__),
        projection_relative_l2=float(np.linalg.norm(residual) / np.linalg.norm(sino)),
        projection_rmse=float(np.sqrt(np.mean(residual**2))),
        reconstruction_min=float(volume.min()),
        reconstruction_max=float(volume.max()),
        units="stored projection units per pixel; physical density conversion unresolved",
        assumptions=[
            "rotation centre detector index 155",
            "circular reconstruction support",
            "skimage angle direction; simulator convention not verified",
        ],
        warning="Full-angle FBP is not ground truth. Residual uses the reconstruction operator; not independent physics validation.",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(
        args.output / "arrays.npz",
        sinogram=sino,
        reconstruction=volume,
        reprojection=predicted,
        residual=residual,
        angles_deg=angles,
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    panels = [
        (phantom, "Known disk", 0, 1),
        (recovered, "Analytic sinogram FBP", 0, 1),
        (volume, "Worm FBP (not ground truth)", float(volume.min()), float(volume.max())),
        (sino, "Clean input sinogram", 0, float(sino.max())),
        (predicted, "FBP reprojection", 0, float(sino.max())),
        (
            residual,
            "Signed projection residual",
            -float(abs(residual).max()),
            float(abs(residual).max()),
        ),
    ]
    for ax, (array, title, lo, hi) in zip(axes.flat, panels):
        im = ax.imshow(
            array,
            cmap="coolwarm" if "residual" in title else "gray",
            vmin=lo,
            vmax=hi,
            aspect="equal" if array.shape[0] == array.shape[1] else "auto",
        )
        ax.set_title(title)
        ax.set_xlabel("pixel / detector index")
        ax.set_ylabel("row / angle index")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(args.output / "diagnostic.png", dpi=150)
    plt.close(fig)
    report["artifacts"] = {
        name: sha256(args.output / name) for name in ["arrays.npz", "diagnostic.png"]
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
