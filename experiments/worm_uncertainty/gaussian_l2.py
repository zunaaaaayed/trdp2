"""Exact full-space L2 for finite mixtures of unnormalized 3D Gaussian densities.

Completing the square gives the pairwise integral below. This metric avoids
voxel sampling and finite-field truncation for matched Gaussian phantoms.
"""

import numpy as np
from cholesky_gaussian import geometry


def inner_product(p, q):
    cp, sp, _, ap = geometry(p)
    cq, sq, _, aq = geometry(q)
    summed = sp[:, None] + sq[None, :]
    delta = cp[:, None] - cq[None, :]
    exponent = np.einsum("abi,abij,abj->ab", delta, np.linalg.inv(summed), delta)
    log_factor = 0.5 * (
        np.linalg.slogdet(sp)[1][:, None]
        + np.linalg.slogdet(sq)[1][None, :]
        - np.linalg.slogdet(summed)[1]
    )
    return float(
        np.sum(ap[:, None] * aq[None, :] * (2 * np.pi) ** 1.5 * np.exp(log_factor - 0.5 * exponent))
    )


def relative_l2(p, truth):
    norm = inner_product(truth, truth)
    if norm <= 0:
        raise ValueError("nonzero truth required")
    error = inner_product(p, p) + norm - 2 * inner_product(p, truth)
    if error < -1e-10 * norm:
        raise ValueError("negative squared error beyond roundoff")
    return float(np.sqrt(max(0, error) / norm))
