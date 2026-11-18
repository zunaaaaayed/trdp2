"""Independent synthetic slice objects and proper density/boundary error metrics."""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from experiments.worm_uncertainty.core.voxel_parallel import relative_l2


FAMILIES = ["smooth", "elongated", "shell", "heterogeneous"]


def make_object(seed, family, size=62, spacing=5):
    rng = np.random.default_rng(seed)
    q = (np.arange(size) - (size - 1) / 2) * spacing
    y, x = np.meshgrid(q, q, indexing="ij")
    angle = rng.uniform(0, np.pi)
    cx, cy = rng.uniform(-25, 25, 2)
    u = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
    v = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
    a, b = rng.uniform(30, 50), rng.uniform(60, 95)
    rho = (u / a) ** 2 + (v / b) ** 2
    level = rng.uniform(0.008, 0.018)
    if family == "smooth":
        result = level * np.exp(-rho * 2)
        result += 0.5 * level * np.exp(-((x + 35) ** 2 + (y - 25) ** 2) / (2 * 18**2))
    elif family == "elongated":
        rho = (u / (a * 0.6)) ** 2 + (v / b) ** 2
        result = level * (rho < 1)
        result += 0.4 * level * ((x + 40) ** 2 + (y + 25) ** 2 < 18**2)
    elif family == "shell":
        result = level * ((rho < 1) & (rho > rng.uniform(0.25, 0.5)))
        result += 0.6 * level * ((x - 30) ** 2 + (y - 30) ** 2 < 14**2)
    elif family == "heterogeneous":
        result = level * (rho < 1) * (0.65 + 0.2 * np.cos(u / 8) + 0.15 * np.sin(v / 13))
        result += 0.5 * level * ((abs(u - 10) < 12) & (abs(v + 30) < 15))
    else:
        raise ValueError("unknown family")
    support = x * x + y * y < (size * spacing / 2) ** 2
    return np.where(support, result, 0)


def objects(split):
    return np.stack(
        [make_object(split["seed_base"] + i, FAMILIES[i % 4]) for i in range(split["count"])]
    )


def density_metrics(volume, truth, spacing=5):
    fg = truth > 0.004
    interior = binary_erosion(fg)
    predicted = volume > 0.004
    a = fg & ~binary_erosion(fg)
    b = predicted & ~binary_erosion(predicted)
    boundary = (
        float(
            (distance_transform_edt(~a)[b].mean() + distance_transform_edt(~b)[a].mean())
            * spacing
            / 2
        )
        if a.any() and b.any()
        else None
    )
    return dict(
        density_l2=relative_l2(volume, truth),
        density_rmse=float(np.sqrt(np.mean((volume - truth) ** 2))),
        foreground_l2=relative_l2(volume[fg], truth[fg]),
        boundary_um=boundary,
        boundary_defined=boundary is not None,
        interior_bias=float((volume - truth)[interior].mean()),
        interior_mae=float(abs(volume - truth)[interior].mean()),
        negative_fraction=float(np.mean(volume < 0)),
    )
