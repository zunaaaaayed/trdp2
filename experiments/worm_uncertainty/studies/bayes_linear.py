"""Small exact Gaussian posterior reference; development calibration only."""

from experiments.worm_uncertainty.layout import resource_path

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from skimage.transform import radon, resize
from experiments.worm_uncertainty.studies.check_fbp import plt
from experiments.worm_uncertainty.studies.phantom_variability import make_phantom
from experiments.worm_uncertainty.data.prepare import sha256


def posterior(operator, sigma, tau):
    if sigma <= 0 or tau <= 0 or not np.isfinite([sigma, tau]).all():
        raise ValueError("positive finite noise and prior scales required")
    precision = operator.T @ operator / sigma**2 + np.eye(operator.shape[1]) / tau**2
    factor = cho_factor(precision, lower=True)
    covariance = cho_solve(factor, np.eye(operator.shape[1]))
    gain = cho_solve(factor, operator.T / sigma**2)
    return gain, covariance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    n = 24
    angles = np.arange(100) * 1.8
    yy, xx = np.mgrid[:n, :n]
    support = (yy - n // 2) ** 2 + (xx - n // 2) ** 2 <= (n // 2 - 1) ** 2
    truth = resize(make_phantom()[0], (n, n), order=1, anti_aliasing=True, preserve_range=True)
    truth[~support] = 0
    coords = np.flatnonzero(support)
    x = truth[support]
    foreground = x > 0.01
    operator = np.empty((100 * n, len(coords)))
    for j, coord in enumerate(coords):
        basis = np.zeros((n, n))
        basis.flat[coord] = 1
        operator[:, j] = radon(basis, theta=angles, circle=True, preserve_range=True).T.ravel()
    direct = radon(truth, theta=angles, circle=True, preserve_range=True).T.ravel()
    operator_error = float(np.max(abs(operator @ x - direct)))
    if operator_error > 1e-10:
        raise ValueError("operator consistency failed")
    sigma = 0.05 * float(np.sqrt(np.mean(direct**2)))
    rng = np.random.default_rng(1701)
    noise = rng.normal(0, sigma, (len(direct), 50))
    arrays = dict(truth=truth, support=support, angles_deg=angles)
    rows = []
    baseline_cov = {}
    for tau in [0.3, 1.0, 3.0]:
        previous = None
        for span in [180, 120, 90]:
            mask = np.repeat(angles < span, n)
            a = operator[mask]
            gain, cov = posterior(a, sigma, tau)
            sd = np.sqrt(np.diag(cov))
            if previous is not None and np.min(np.diag(cov) - previous) < -1e-10:
                raise ValueError("uncertainty decreased after removing measurements")
            previous = np.diag(cov)
            means = gain @ (direct[mask, None] + noise[mask])
            errors = means - x[:, None]
            # Independent draws from exactly the assumed prior and likelihood.
            calibration_rng = np.random.default_rng(1900 + span)
            prior_truth = calibration_rng.normal(0, tau, (len(x), 200))
            prior_noise = calibration_rng.normal(0, sigma, (mask.sum(), 200))
            prior_errors = gain @ (a @ prior_truth + prior_noise) - prior_truth
            row = dict(
                tau=tau,
                span=span,
                foreground_rmse=float(np.sqrt(np.mean(errors[foreground] ** 2))),
                mean_posterior_sd=float(sd[foreground].mean()),
                foreground_95_coverage=float(
                    np.mean(abs(errors[foreground]) <= 1.95996398454 * sd[foreground, None])
                ),
                mean_95_interval_width=float((2 * 1.95996398454 * sd[foreground]).mean()),
                prior_predictive_95_coverage=float(
                    np.mean(abs(prior_errors) <= 1.95996398454 * sd[:, None])
                ),
            )
            rows.append(row)
            print(row, flush=True)
            if tau == 1.0:
                for name, v in [("mean", means[:, 0]), ("sd", sd), ("error", abs(errors[:, 0]))]:
                    image = np.zeros((n, n))
                    image[support] = v
                    arrays[f"{name}_{span}"] = image
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    fig, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
    error_max = max(arrays[f"error_{s}"].max() for s in [180, 120, 90])
    sd_max = max(arrays[f"sd_{s}"].max() for s in [180, 120, 90])
    for i, span in enumerate([180, 120, 90]):
        panels = [
            (truth, "Known density\nbrightness is not quality", -0.2, 1.2, "gray"),
            (
                arrays[f"mean_{span}"],
                "Reconstruction\ncloser to truth is better",
                -0.2,
                1.2,
                "gray",
            ),
            (
                arrays[f"error_{span}"],
                "Absolute error\n0 is ideal; higher is worse",
                0,
                error_max,
                "magma",
            ),
            (arrays[f"sd_{span}"], "Posterior SD\nhigher = less certain", 0, sd_max, "viridis"),
        ]
        for j, (v, title, lo, hi, cmap) in enumerate(panels):
            im = axes[i, j].imshow(v, vmin=lo, vmax=hi, cmap=cmap)
            axes[i, j].set_title(f"{span}°: {title}", fontsize=9)
            axes[i, j].set_axis_off()
            if i == 2:
                fig.colorbar(
                    im, ax=axes[:, j].tolist(), shrink=0.55, label="arbitrary density units"
                )
    fig.suptitle(
        "Exact linear Gaussian reference: prior SD = 1; first noise draw\nUncertainty quality requires coverage AND interval width; low SD alone is not success.",
        fontsize=11,
    )
    fig.savefig(args.output / "posterior.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for tau in [0.3, 1.0, 3.0]:
        selected = [r for r in rows if r["tau"] == tau]
        ax.plot(
            [r["span"] for r in selected],
            [100 * r["foreground_95_coverage"] for r in selected],
            "o-",
            label=f"Prior SD {tau:g}",
        )
    ax.axhline(95, color="black", linestyle="--", label="Nominal target: 95%")
    ax.set(
        xlabel="Angular coverage (degrees)",
        ylabel="Empirical foreground coverage (%)",
        ylim=(0, 101),
        title="Closer to 95% is better calibrated; above 95% can mean overly wide intervals",
    )
    ax.legend()
    fig.savefig(args.output / "coverage.png", dpi=140)
    plt.close(fig)
    report = dict(
        kind="exact_linear_gaussian_development_reference",
        shape=[n, n],
        unknowns=len(x),
        sigma=sigma,
        noise_draws=50,
        prior_predictive_draws=200,
        noise_seed=1701,
        prior="zero-mean independent Gaussian; no positivity constraint",
        prior_scales=[0.3, 1.0, 3.0],
        operator_max_error=operator_error,
        rows=rows,
        code_sha256={
            name: sha256(resource_path(name))
            for name in ["bayes_linear.py", "phantom_variability.py"]
        },
        limitations=[
            "single resized development phantom, matched projection operator",
            "prior scales are sensitivity probes, not selected using these results",
            "coverage is fixed-truth repeated-noise coverage; not a universal posterior guarantee",
            "pixels correlated; no pixel-independent significance test",
            "coarser grid and different prior than SART; not a performance comparison",
            "Gaussian prior allows negative densities; not a final worm model",
        ],
        artifacts={
            name: sha256(args.output / name)
            for name in ["arrays.npz", "posterior.png", "coverage.png"]
        },
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
