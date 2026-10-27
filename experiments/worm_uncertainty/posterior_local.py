"""Local Gauss-Newton Gaussian covariance and mean-only Gaussian evaluators."""

import numpy as np
from scipy.linalg import solve_triangular
from cholesky_gaussian import covariance_matrix


def gaussian_factor(jacobian, prior_sd, sigma):
    """Return physical-coordinate covariance factor; C=F F.T, not inverse Hessian squared."""
    a = jacobian * prior_sd[None, :] / sigma
    precision = a.T @ a + np.eye(len(prior_sd))
    lower = np.linalg.cholesky(precision)
    factor = prior_sd[:, None] * solve_triangular(lower.T, np.eye(len(prior_sd)), lower=False)
    return factor, float(np.linalg.cond(precision))


def mean_geometry(p):
    k = len(p) // 10
    return (
        p[: 3 * k].reshape(k, 3),
        covariance_matrix(p[3 * k : 6 * k].reshape(k, 3), p[6 * k : 9 * k].reshape(k, 3)),
        p[9 * k :],
    )


def projection_mean(p, angles, uv):
    centres, cov, density = mean_geometry(p)
    e = np.zeros((len(angles), 2, 3))
    e[:, 0, 0] = np.cos(angles)
    e[:, 0, 1] = -np.sin(angles)
    e[:, 1, 2] = 1
    projected = np.einsum("aij,kjl,aml->akim", e, cov, e)
    delta = uv[None, :, None, :] - np.einsum("aij,kj->aki", e, centres)[:, None, :, :]
    exponent = np.einsum("anki,akij,ankj->ank", delta, np.linalg.inv(projected), delta)
    basis = np.sqrt(2 * np.pi * np.linalg.det(cov)[None, :] / np.linalg.det(projected))[
        :, None, :
    ] * np.exp(-0.5 * exponent)
    return (basis @ density).ravel()


def voxel_mean(p, points):
    centres, cov, density = mean_geometry(p)
    delta = points[:, None, :] - centres[None, :, :]
    exponent = np.einsum("nki,kij,nkj->nk", delta, np.linalg.inv(cov), delta)
    return np.exp(-0.5 * exponent) @ density


def interval_metrics(samples, truth, mask):
    if not np.isfinite(samples).all():
        raise ValueError("nonfinite propagated sample")
    lo, hi = np.quantile(samples, [0.025, 0.975], axis=0)
    mean = np.mean(samples, axis=0)
    if not np.any(mask):
        raise ValueError("empty evaluation mask")
    return (
        dict(
            coverage95=float(np.mean((truth[mask] >= lo[mask]) & (truth[mask] <= hi[mask]))),
            mean_width95=float(np.mean((hi - lo)[mask])),
            mean_rmse=float(np.sqrt(np.mean((mean[mask] - truth[mask]) ** 2))),
        ),
        mean,
        lo,
        hi,
    )
