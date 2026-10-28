"""Known-target development check of noise variability versus reconstruction bias."""

from experiments.worm_uncertainty.layout import resource_path

import argparse
import json
from pathlib import Path
import numpy as np
import skimage
from skimage.transform import radon
from experiments.worm_uncertainty.studies.compare_sart import fit_sart
from experiments.worm_uncertainty.studies.check_fbp import plt
from experiments.worm_uncertainty.data.prepare import angle_mask, noisy_copy, sha256


def make_phantom():
    y, x = np.mgrid[:64, :64]
    support = ((x - 32) / 23) ** 2 + ((y - 32) / 26) ** 2 <= 1
    truth = np.zeros((64, 64))
    truth[support] = 0.3
    truth[((x - 23) / 7) ** 2 + ((y - 25) / 12) ** 2 <= 1] = 0.75
    truth[((x - 42) / 5) ** 2 + ((y - 36) / 10) ** 2 <= 1] = 1.0
    truth[(x - 29) ** 2 + (y - 45) ** 2 <= 3**2] = 1.2
    truth[(x - 39) ** 2 + (y - 19) ** 2 <= 2**2] = 0.1
    # A textured interior patch, fully inside the object.
    patch = (x >= 28) & (x <= 35) & (y >= 22) & (y <= 35)
    truth[patch] = 0.45 + 0.1 * np.cos((x[patch] - 28) * np.pi / 2)
    return truth, support, patch


def summarize(stack, truth, mask):
    mean = stack.mean(axis=0)
    std = stack.std(axis=0, ddof=1)
    bias = mean - truth
    mse = float(np.mean((stack[:, mask] - truth[mask]) ** 2))
    bias2 = float(np.mean(bias[mask] ** 2))
    variance = float(np.mean(stack.var(axis=0, ddof=0)[mask]))
    return dict(
        rmse=float(np.sqrt(mse)),
        mean_bias_rmse=float(np.sqrt(bias2)),
        mean_noise_std=float(std[mask].mean()),
        squared_bias_fraction_of_mse=bias2 / mse if mse else 0.0,
        bias_variance_identity_error=abs(mse - bias2 - variance),
        fraction_bias_exceeding_two_noise_std=float(np.mean(abs(bias[mask]) > 2 * std[mask])),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    truth, support, patch = make_phantom()
    angles = np.arange(100) * 1.8
    clean = radon(truth, theta=angles, circle=True, preserve_range=True).T
    scale = float(np.sqrt(np.mean(clean**2)))
    seeds = list(range(20))
    fraction = 0.05
    # Same complete noise realization is shared by all angular masks.
    noisy = [noisy_copy(clean, scale, fraction, seed) for seed in seeds]
    gy, gx = np.gradient(truth)
    boundary = support & (np.hypot(gx, gy) > 0)
    regions = dict(
        foreground=support, boundary=boundary, interior=support & ~boundary, texture=patch
    )
    arrays = dict(truth=truth, clean=clean, angles_deg=angles, **regions)
    rows = []
    for span in [180, 120, 90]:
        _, mask = angle_mask(0, span)
        baseline = fit_sart(clean, angles, mask)
        stack = np.stack([fit_sart(v, angles, mask) for v in noisy])
        arrays[f"clean_reconstruction_{span}"] = baseline
        arrays[f"reconstructions_{span}"] = stack
        row = dict(
            span=span,
            measured_views=int(mask.sum()),
            clean_foreground_rmse=float(
                np.sqrt(np.mean((baseline[support] - truth[support]) ** 2))
            ),
            regions={name: summarize(stack, truth, region) for name, region in regions.items()},
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    max_error = max(
        abs(arrays[f"reconstructions_{s}"].mean(axis=0) - truth).max() for s in [180, 120, 90]
    )
    # Error and variability share the same scale, so their magnitudes are comparable.
    for i, span in enumerate([180, 120, 90]):
        stack = arrays[f"reconstructions_{span}"]
        mean = stack.mean(axis=0)
        images = [truth, mean, abs(mean - truth), stack.std(axis=0, ddof=1)]
        titles = [
            "Ground truth",
            "Mean reconstruction",
            "Absolute bias",
            "Noise standard deviation",
        ]
        for j, (a, title) in enumerate(zip(images, titles)):
            im = axes[i, j].imshow(
                a, cmap="gray" if j < 2 else "magma", vmin=0, vmax=1.2 if j < 2 else max_error
            )
            axes[i, j].set_title(f"{span}°: {title}", fontsize=9)
            axes[i, j].set_axis_off()
            if i == 2:
                fig.colorbar(im, ax=axes[:, j].tolist(), shrink=0.6)
    fig.savefig(args.output / "comparison.png", dpi=150)
    plt.close(fig)
    report = dict(
        kind="known_target_noise_variability_diagnostic",
        shape=[64, 64],
        seeds=seeds,
        noise_fraction=fraction,
        signal_scale=scale,
        sigma=scale * fraction,
        reconstruction="nonnegative circular-support SART, zero init, 10 sweeps, relaxation 0.15",
        metrics=rows,
        numpy_version=np.__version__,
        skimage_version=skimage.__version__,
        limitations=[
            "one development phantom; matched numerical projection model",
            "noise response is not posterior uncertainty",
            "bias-exceeds-2std is descriptive, not confidence interval coverage",
            "20 noise draws; spatial pixels are correlated",
            "arbitrary density units, no physical calibration",
        ],
        code_sha256={
            name: sha256(resource_path(name))
            for name in ["phantom_variability.py", "compare_sart.py", "prepare.py"]
        },
        artifacts={name: sha256(args.output / name) for name in ["arrays.npz", "comparison.png"]},
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
