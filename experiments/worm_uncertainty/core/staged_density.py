"""Bounded conditional density solve and shared-objective staged optimization.

Only acquired noisy rays enter these functions. A conditional convex optimum
neither identifies geometry nor establishes missing-view accuracy.
"""

import time

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from experiments.worm_uncertainty.core.cholesky_gaussian import projection


def prior_and_bounds(reference, cfg):
    k = len(reference) // 10
    mean = reference.copy()
    mean[9 * k :] = 0
    sd = np.repeat(
        [
            cfg["position_prior_sd"],
            cfg["cholesky_log_diagonal_prior_sd"],
            cfg["cholesky_off_diagonal_prior_sd"],
            cfg["density_prior_sd"],
        ],
        [3 * k, 3 * k, 3 * k, k],
    )
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
    return mean, sd, lower, upper


def solve_density(basis, y, sigma, tau, bounds):
    """Solve bounded ridge; report KKT residual scaled by gradient term magnitudes."""
    basis, y = np.asarray(basis), np.asarray(y)
    if sigma <= 0 or tau <= 0 or basis.ndim != 2 or y.shape != (len(basis),):
        raise ValueError("positive scales and compatible matrix/vector required")
    if not np.isfinite(basis).all() or not np.isfinite(y).all():
        raise ValueError("finite inputs required")
    matrix = np.vstack([basis / sigma, np.eye(basis.shape[1]) / tau])
    target = np.r_[y / sigma, np.zeros(basis.shape[1])]
    result = lsq_linear(matrix, target, bounds=bounds, method="bvls", tol=1e-12, max_iter=1000)
    gradient = matrix.T @ (matrix @ result.x - target)
    violation = gradient.copy()
    violation[result.active_mask == -1] = np.minimum(gradient[result.active_mask == -1], 0)
    violation[result.active_mask == 1] = np.maximum(gradient[result.active_mask == 1], 0)
    scale = max(
        1.0,
        np.linalg.norm(matrix.T @ target, ord=np.inf),
        np.linalg.norm(matrix.T @ (matrix @ result.x), ord=np.inf),
    )
    kkt = float(np.linalg.norm(violation, ord=np.inf))
    info = dict(
        success=bool(result.success),
        iterations=int(result.nit),
        kkt_absolute=kkt,
        kkt_relative=kkt / scale,
        active_indices=np.flatnonzero(result.active_mask).tolist(),
        augmented_condition=float(np.linalg.cond(matrix)),
    )
    if not result.success or info["kkt_relative"] > 1e-9:
        raise RuntimeError("conditional density solve failed KKT check")
    return result.x, info


def fit_stages(y, angles, uv, initial, reference, cfg, stages):
    """Fixed stage schedule; final joint solve supplies termination and gate inputs.

    Each stage minimizes the same sum-of-squares objective. Block termination alone
    is never reported as convergence of the full model.
    """
    started = time.monotonic()
    mean, sd, lower, upper = prior_and_bounds(reference, cfg)
    k, sigma = len(initial) // 10, cfg["noise_sigma"]
    p = initial.copy()
    count = 0
    cache = {}

    def values(x):
        nonlocal count
        if "x" not in cache or not np.array_equal(cache["x"], x):
            cache["x"] = x.copy()
            cache["v"] = projection(x, angles, uv)
            count += 1
        return cache["v"]

    def residual(x):
        return np.r_[(values(x)[0] - y) / sigma, (x - mean) / sd]

    def terms(x):
        r = residual(x)
        return dict(
            data_term=float(r[: len(y)] @ r[: len(y)]),
            prior_term=float(r[len(y) :] @ r[len(y) :]),
            objective=float(r @ r),
        )

    trace = []
    final = None
    for stage in stages:
        before = terms(p)
        if stage["kind"] == "density":
            p[9 * k :], info = solve_density(
                values(p)[1][:, 9 * k :], y, sigma, cfg["density_prior_sd"], cfg["density_bounds"]
            )
        else:
            size = {"centre": 3 * k, "geometry": 9 * k, "joint": 10 * k}[stage["kind"]]
            base = p.copy()

            def expand(x):
                full = base.copy()
                full[:size] = x
                return full

            def jac(x):
                return np.vstack([values(expand(x))[1] / sigma, np.diag(1 / sd)])[:, :size]

            result = least_squares(
                lambda x: residual(expand(x)),
                p[:size],
                jac=jac,
                bounds=(lower[:size], upper[:size]),
                x_scale="jac",
                max_nfev=stage["budget"],
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            p = expand(result.x)
            info = dict(
                success=bool(result.success),
                nfev=int(result.nfev),
                optimality=float(result.optimality),
                status=int(result.status),
            )
            if stage["kind"] == "joint":
                final = result
        after = terms(p)
        if after["objective"] > before["objective"] + 1e-8 * max(1, before["objective"]):
            raise RuntimeError("stage increased the shared objective")
        trace.append(
            dict(
                stage=stage,
                before=before,
                after=after,
                solver=info,
                projection_jacobian_calls=count,
            )
        )
    if final is None or stages[-1]["kind"] != "joint":
        raise ValueError("final joint solve required for full-model convergence status")
    return dict(
        parameters=p.tolist(),
        **terms(p),
        converged=bool(final.success),
        optimality=float(final.optimality),
        active_indices=np.flatnonzero(final.active_mask).tolist(),
        nfev=sum(t["solver"].get("nfev", 0) for t in trace),
        projection_jacobian_calls=count,
        trace=trace,
        elapsed_seconds=time.monotonic() - started,
    )
