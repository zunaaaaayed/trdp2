"""Initialize Gaussian components using only the supplied acquired projections.

Greedy nonnegative dictionary fitting provides centres and amplitudes. Widths
are a fixed design choice; no truth geometry or withheld rays are accepted.
"""

import numpy as np
from scipy.optimize import nnls
from experiments.worm_uncertainty.studies.diagnose_3d import ray_basis


def dictionary(angles, uv, cfg):
    xy = np.linspace(
        -cfg["initialization_extent"], cfg["initialization_extent"], cfg["initialization_grid_xy"]
    )
    z = np.linspace(
        -cfg["initialization_extent"], cfg["initialization_extent"], cfg["initialization_grid_z"]
    )
    centres = np.stack(np.meshgrid(xy, xy, z, indexing="ij"), axis=-1).reshape(-1, 3)
    widths = np.full_like(centres, cfg["initialization_width"])
    basis = ray_basis(centres, widths, angles, uv)[0]
    norms = np.linalg.norm(basis, axis=0)
    return centres, basis, norms


def initialize(acquired_y, angles, uv, cfg, prepared=None):
    y = np.asarray(acquired_y).reshape(-1)
    if len(y) != len(angles) * len(uv) or not np.isfinite(y).all():
        raise ValueError("finite acquired projections with matching shape required")
    centres, basis, norms = dictionary(angles, uv, cfg) if prepared is None else prepared
    selected = []
    allowed = norms > 1e-12
    residual = y.copy()
    for _ in range(cfg["components"]):
        score = np.where(allowed, basis.T @ residual / np.maximum(norms, 1e-12), -np.inf)
        chosen = int(np.argmax(score))
        if not np.isfinite(score[chosen]):
            raise ValueError("no remaining dictionary atom")
        selected.append(chosen)
        coefficients = nnls(basis[:, selected], y)[0]
        residual = y - basis[:, selected] @ coefficients
        allowed &= (
            np.linalg.norm(centres - centres[chosen], axis=1) >= cfg["minimum_centre_distance"]
        )
        allowed[chosen] = False
    density = np.clip(coefficients, 0.01, cfg["density_bounds"][1] - 0.01)
    k = len(selected)
    parameters = np.r_[
        centres[selected].ravel(),
        np.full(3 * k, np.log(cfg["initialization_width"])),
        np.zeros(3 * k),
        density,
    ]
    return parameters, dict(
        selected_indices=selected,
        acquired_relative_residual=float(np.linalg.norm(residual) / max(np.linalg.norm(y), 1e-15)),
        density_clipped=bool(np.any(density != coefficients)),
    )


def perturb(reference, seed, cfg):
    if seed == 0:
        return reference.copy()
    rng = np.random.default_rng(seed)
    k = len(reference) // 10
    p = reference.copy()
    p[: 3 * k] += rng.normal(0, cfg["start_centre_sd"], 3 * k)
    p[: 3 * k] = np.clip(
        p[: 3 * k], cfg["position_bounds"][0] + 0.01, cfg["position_bounds"][1] - 0.01
    )
    p[3 * k : 6 * k] += rng.normal(0, cfg["start_log_diagonal_sd"], 3 * k)
    p[6 * k : 9 * k] += rng.normal(0, cfg["start_off_diagonal_sd"], 3 * k)
    p[9 * k :] *= np.exp(rng.normal(0, cfg["start_density_fraction_sd"], k))
    p[9 * k :] = np.clip(p[9 * k :], 0.01, cfg["density_bounds"][1] - 0.01)
    return p
