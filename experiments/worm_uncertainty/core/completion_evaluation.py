"""Acquired-only fit selection and diagnostic reconstruction/Fourier views."""

import numpy as np
from experiments.worm_uncertainty.core.cholesky_gaussian import geometry, projection


def common_starts(fits, arms):
    return sorted(
        set.intersection(
            *[{r["seed"] for r in fits if r["arm"] == a and "parameters" in r} for a in arms]
        )
    )


def select_fit(fits, reference, cfg, angles, uv):
    from experiments.worm_uncertainty.core.staged_density import prior_and_bounds

    _, _, lo, hi = prior_and_bounds(reference, cfg)
    valid = [
        r
        for r in fits
        if "parameters" in r
        and np.isfinite(r["objective"])
        and np.isfinite(r["parameters"]).all()
        and np.all(np.asarray(r["parameters"]) >= lo)
        and np.all(np.asarray(r["parameters"]) <= hi)
    ]
    accepted = [r for r in valid if r["converged"] and r["numerical"]["accepted"]]
    if not valid:
        return None, dict(passed=False, reason="no finite feasible candidates")
    chosen = min(accepted or valid, key=lambda r: r["objective"])
    gate = dict(passed=False, accepted_candidates=len(accepted), provisional=not bool(accepted))
    thresholds = cfg["stability_gate"]
    if len(accepted) >= thresholds["minimum_converged"]:
        first, second = sorted(accepted, key=lambda r: r["objective"])[:2]
        a = projection(np.asarray(first["parameters"]), angles, uv)[0]
        b = projection(np.asarray(second["parameters"]), angles, uv)[0]
        gap = abs(first["objective"] - second["objective"]) / max(1, abs(first["objective"]))
        difference = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-15))
        gate.update(
            relative_objective_gap=gap,
            relative_prediction_disagreement=difference,
            passed=not (first["active_indices"] or second["active_indices"])
            and gap <= thresholds["relative_objective_gap"]
            and difference <= thresholds["relative_prediction_disagreement"],
        )
    return chosen, gate


def filtered_backprojection(sinogram, u, angles, side=96):
    """Ramp-filter diagnostic, consistent u=x cos(theta)-y sin(theta).

    Clean full-angle FBP is a surrogate, not a density ground-truth volume.
    Point sampling and finite detector support can produce artifacts.
    """
    n = sinogram.shape[1]
    du = float(u[1] - u[0])
    size = 2 ** int(np.ceil(np.log2(2 * n)))
    frequencies = np.fft.rfftfreq(size, du)
    filtered = np.fft.irfft(
        np.fft.rfft(sinogram, n=size, axis=1) * frequencies[None, :], n=size, axis=1
    )[:, :n]
    xy = np.linspace(-1, 1, side)
    xx, yy = np.meshgrid(xy, xy, indexing="xy")
    image = np.zeros_like(xx)
    for angle, row in zip(angles, filtered):
        image += np.interp(xx * np.cos(angle) - yy * np.sin(angle), u, row, left=0, right=0)
    return image * np.pi / len(angles)


def density_slice(p, z, side=96):
    centres, cov, _, density = geometry(p)
    xy = np.linspace(-1, 1, side)
    xx, yy = np.meshgrid(xy, xy, indexing="xy")
    xyz = np.stack([xx, yy, np.full_like(xx, z)], axis=-1)
    delta = xyz[:, :, None, :] - centres
    return np.sum(
        density[None, None, :]
        * np.exp(-0.5 * np.einsum("xyki,kij,xykj->xyk", delta, np.linalg.inv(cov), delta)),
        axis=-1,
    )


def polar_spectrum(sinogram, u, angles):
    freq = np.fft.fftshift(np.fft.fftfreq(len(u), float(u[1] - u[0])))
    magnitude = np.abs(np.fft.fftshift(np.fft.fft(sinogram, axis=1), axes=1)) * (u[1] - u[0])
    return (
        freq[None, :] * np.cos(angles[:, None]),
        -freq[None, :] * np.sin(angles[:, None]),
        np.log1p(magnitude),
    )
