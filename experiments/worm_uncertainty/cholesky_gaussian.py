"""Positive-definite covariance coordinates for a shared-prior optimization control.

Shape blocks: log Cholesky diagonals, then raw lower entries (10,20,21).
This defines a different shape prior from the Euler experiment.
"""

import numpy as np
from scipy.optimize import least_squares
from shape_gaussian import covariances
from rotation_sensitivity import initial_parameters as euler_initial


def convert_euler(p):
    k = len(p) // 10
    covariance = covariances(p[3 * k : 6 * k].reshape(k, 3), p[6 * k : 9 * k].reshape(k, 3))
    lower = np.linalg.cholesky(covariance)
    return np.r_[
        p[: 3 * k],
        np.log(np.diagonal(lower, axis1=1, axis2=2)).ravel(),
        lower[:, [1, 2, 2], [0, 0, 1]].ravel(),
        p[9 * k :],
    ]


def initial_parameters(centres, scales, density, rotations, seed, cfg):
    return convert_euler(euler_initial(centres, scales, density, rotations, seed, cfg))


def covariance_matrix(diagonals, off):
    k = len(diagonals)
    lower = np.zeros((k, 3, 3), dtype=np.result_type(diagonals, off))
    lower[:, np.arange(3), np.arange(3)] = np.exp(diagonals)
    lower[:, [1, 2, 2], [0, 0, 1]] = off
    return lower @ lower.transpose(0, 2, 1)


def geometry(p):
    k = len(p) // 10
    diagonals = p[3 * k : 6 * k].reshape(k, 3)
    off = p[6 * k : 9 * k].reshape(k, 3)
    derivatives = []
    for j in range(6):
        d = diagonals.astype(complex)
        o = off.astype(complex)
        (d if j < 3 else o)[:, j % 3] += 1e-20j
        derivatives.append(np.imag(covariance_matrix(d, o)) / 1e-20)
    return (
        p[: 3 * k].reshape(k, 3),
        covariance_matrix(diagonals, off),
        np.stack(derivatives, axis=1),
        p[9 * k :],
    )


def fit_control(y, angles, uv, initial, prior_reference, sigma, cfg, bounded):
    if bounded:
        raise ValueError("Cholesky control has no shape-coordinate bounds")
    k = len(initial) // 10
    prior_mean = prior_reference.copy()
    prior_mean[9 * k :] = 0
    sd = np.r_[
        np.full(3 * k, cfg["position_prior_sd"]),
        np.full(3 * k, cfg["cholesky_log_diagonal_prior_sd"]),
        np.full(3 * k, cfg["cholesky_off_diagonal_prior_sd"]),
        np.full(k, cfg["density_prior_sd"]),
    ]
    cache = {}

    def values(p):
        if "p" not in cache or not np.array_equal(cache["p"], p):
            cache["p"] = p.copy()
            cache["values"] = projection(p, angles, uv)
        return cache["values"]

    def residual(p):
        return np.r_[(values(p)[0] - y) / sigma, (p - prior_mean) / sd]

    def jacobian(p):
        return np.vstack([values(p)[1] / sigma, np.diag(1 / sd)])

    lower = np.r_[
        np.full(3 * k, cfg["position_bounds"][0]),
        np.full(6 * k, -np.inf),
        np.full(k, cfg["density_bounds"][0]),
    ]
    upper = np.r_[
        np.full(3 * k, cfg["position_bounds"][1]),
        np.full(6 * k, np.inf),
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
        x_scale="jac",
    )
    return fit, sd


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
