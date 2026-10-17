"""Analytic parallel-beam Gaussian features with centre, width and rotation derivatives.

Parameters are component-major within four blocks: centres, log standard
 deviations, XYZ Euler angles (Rz @ Ry @ Rx), then peak densities.
"""

import numpy as np
from scipy.optimize import least_squares
from diagnose_3d import invert_precision
from stress_3d import propagated_sd


def covariances(log_widths, rotations):
    x, y, z = rotations.T
    zeros = np.zeros_like(x)
    ones = np.ones_like(x)

    def matrix(rows):
        return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)

    rx = matrix(
        [[ones, zeros, zeros], [zeros, np.cos(x), -np.sin(x)], [zeros, np.sin(x), np.cos(x)]]
    )
    ry = matrix(
        [[np.cos(y), zeros, np.sin(y)], [zeros, ones, zeros], [-np.sin(y), zeros, np.cos(y)]]
    )
    rz = matrix(
        [[np.cos(z), -np.sin(z), zeros], [np.sin(z), np.cos(z), zeros], [zeros, zeros, ones]]
    )
    rotation = rz @ ry @ rx
    return (rotation * np.exp(2 * log_widths)[:, None, :]) @ rotation.transpose(0, 2, 1)


def geometry(parameters):
    k = len(parameters) // 10
    centres = parameters[: 3 * k].reshape(k, 3)
    widths = parameters[3 * k : 6 * k].reshape(k, 3)
    rotations = parameters[6 * k : 9 * k].reshape(k, 3)
    sigma = covariances(widths, rotations)
    derivatives = []
    # Complex-step only the small covariance matrices; ray derivatives below
    # are analytic. Each component depends only on its own shape parameters.
    for j in range(6):
        w = widths.astype(complex)
        r = rotations.astype(complex)
        (w if j < 3 else r)[:, j % 3] += 1e-20j
        derivatives.append(np.imag(covariances(w, r)) / 1e-20)
    return centres, sigma, np.stack(derivatives, axis=1), parameters[9 * k :]


def projection(parameters, angles, uv):
    centres, sigma, ds, density = geometry(parameters)
    e = np.zeros((len(angles), 2, 3))
    e[:, 0, 0] = np.cos(angles)
    e[:, 0, 1] = -np.sin(angles)
    e[:, 1, 2] = 1
    s = np.einsum("aij,kjl,aml->akim", e, sigma, e)
    inv = np.linalg.inv(s)
    delta = uv[None, :, None, :] - np.einsum("aij,kj->aki", e, centres)[:, None, :, :]
    w = np.einsum("akij,ankj->anki", inv, delta)
    basis = np.sqrt(2 * np.pi * np.linalg.det(sigma)[None, :] / np.linalg.det(s))[
        :, None, :
    ] * np.exp(-0.5 * np.sum(delta * w, axis=-1))
    centre_derivative = np.einsum("anki,aij->ankj", w, e)
    ds_projected = np.einsum("aij,kbjl,aml->akbim", e, ds, e)
    trace_full = np.einsum("kij,kbji->kb", np.linalg.inv(sigma), ds)
    trace_projected = np.einsum("akij,akbji->akb", inv, ds_projected)
    shape_derivative = 0.5 * (
        trace_full[None, None, :, :]
        - trace_projected[:, None, :, :]
        + np.einsum("anki,akbij,ankj->ankb", w, ds_projected, w)
    )
    weighted = basis * density[None, None, :]
    n = basis.shape[0] * basis.shape[1]
    jac = np.concatenate(
        [
            (centre_derivative * weighted[:, :, :, None]).reshape(n, -1),
            (shape_derivative[:, :, :, :3] * weighted[:, :, :, None]).reshape(n, -1),
            (shape_derivative[:, :, :, 3:] * weighted[:, :, :, None]).reshape(n, -1),
            basis.reshape(n, -1),
        ],
        axis=1,
    )
    return (basis @ density).ravel(), jac


def voxel(parameters, points):
    centres, sigma, ds, density = geometry(parameters)
    delta = points[:, None, :] - centres[None, :, :]
    w = np.einsum("kij,nkj->nki", np.linalg.inv(sigma), delta)
    basis = np.exp(-0.5 * np.sum(delta * w, axis=-1))
    shape = 0.5 * np.einsum("nki,kbij,nkj->nkb", w, ds, w)
    weighted = basis * density[None, :]
    jac = np.concatenate(
        [
            (w * weighted[:, :, None]).reshape(len(points), -1),
            (shape[:, :, :3] * weighted[:, :, None]).reshape(len(points), -1),
            (shape[:, :, 3:] * weighted[:, :, None]).reshape(len(points), -1),
            basis,
        ],
        axis=1,
    )
    return basis @ density, jac


def fit_shape(y, angles, uv, initial, sigma, cfg):
    k = len(initial) // 10
    prior_mean = initial.copy()
    prior_mean[9 * k :] = 0
    prior_sd = np.r_[
        np.full(3 * k, cfg["position_prior_sd"]),
        np.full(3 * k, cfg["log_width_prior_sd"]),
        np.full(3 * k, cfg["rotation_prior_sd"]),
        np.full(k, cfg["density_prior_sd"]),
    ]
    # Cache paired residual/Jacobian evaluations at the same optimizer point.
    cache = {}

    def values(p):
        if "p" not in cache or not np.array_equal(cache["p"], p):
            cache["p"] = p.copy()
            cache["values"] = projection(p, angles, uv)
        return cache["values"]

    def residual(p):
        return np.r_[(values(p)[0] - y) / sigma, (p - prior_mean) / prior_sd]

    def jacobian(p):
        return np.vstack([values(p)[1] / sigma, np.diag(1 / prior_sd)])

    lower = np.r_[
        np.full(3 * k, cfg["position_bounds"][0]),
        np.full(3 * k, np.log(cfg["width_bounds"][0])),
        np.full(3 * k, -cfg["rotation_bound"]),
        np.full(k, cfg["density_bounds"][0]),
    ]
    upper = np.r_[
        np.full(3 * k, cfg["position_bounds"][1]),
        np.full(3 * k, np.log(cfg["width_bounds"][1])),
        np.full(3 * k, cfg["rotation_bound"]),
        np.full(k, cfg["density_bounds"][1]),
    ]
    fit = least_squares(
        residual,
        initial,
        jac=jacobian,
        bounds=(lower, upper),
        max_nfev=cfg["budget_nfev"],
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    return fit, prior_sd


def uncertainty(jac, voxel_jac, measured, sigma, prior_sd, k):
    precision = jac[measured].T @ jac[measured] / sigma**2 + np.diag(1 / prior_sd**2)
    blocks = {
        "density": np.arange(9 * k, 10 * k),
        "centres": np.r_[0 : 3 * k, 9 * k : 10 * k],
        "widths": np.r_[0 : 6 * k, 9 * k : 10 * k],
        "rotations": np.arange(10 * k),
    }
    result = {}
    for name, indices in blocks.items():
        covariance = invert_precision(precision[np.ix_(indices, indices)])
        result[name] = (
            propagated_sd(jac[:, indices], covariance),
            propagated_sd(voxel_jac[:, indices], covariance),
        )
    return result
