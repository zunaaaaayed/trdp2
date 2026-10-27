"""Dimensionless local numerical acceptance; not an accuracy or posterior gate."""

import numpy as np
from scipy.optimize import lsq_linear

from cholesky_gaussian import projection
from staged_density import prior_and_bounds


def assess_linearization(p, residual, jacobian, sd, lower, upper, thresholds):
    """Invariant under positive diagonal changes of parameter units.

    Uses the full data-plus-prior residual. Feasible KKT signs handle active bounds,
    which are also reported separately for downstream posterior eligibility.
    """
    arrays = [p, residual, jacobian, sd]
    if not all(np.isfinite(a).all() for a in arrays) or np.any(sd <= 0):
        raise ValueError("finite inputs and positive parameter scales required")
    if np.any(p < lower) or np.any(p > upper):
        raise ValueError("infeasible parameters")
    a = jacobian * sd[None, :]
    gradient = a.T @ residual
    lo, hi = (lower - p) / sd, (upper - p) / sd
    lower_active = -lo <= thresholds["bound_tolerance"]
    upper_active = hi <= thresholds["bound_tolerance"]
    violation = gradient.copy()
    violation[lower_active] = np.minimum(violation[lower_active], 0)
    violation[upper_active] = np.maximum(violation[upper_active], 0)
    norms = np.linalg.norm(a, axis=0)
    if np.any(norms == 0):
        raise ValueError("nonzero augmented Jacobian columns required")
    scaled_gradient = float(np.max(abs(violation) / norms) / max(1.0, np.linalg.norm(residual)))
    step = lsq_linear(a, -residual, bounds=(lo, hi), method="bvls", tol=1e-12, max_iter=300)
    objective = float(residual @ residual)
    linear = residual + a @ step.x
    reduction = max(0.0, objective - float(linear @ linear)) / max(1.0, objective)
    correction = float(np.linalg.norm(step.x, ord=np.inf))
    accepted = bool(
        step.success
        and scaled_gradient <= thresholds["normalized_gradient"]
        and reduction <= thresholds["relative_predicted_decrease"]
        and correction <= thresholds["prior_scaled_correction_inf"]
    )
    return dict(
        accepted=accepted,
        normalized_gradient=scaled_gradient,
        relative_predicted_decrease=reduction,
        prior_scaled_correction_inf=correction,
        step_solver_success=bool(step.success),
        step_solver_optimality=float(step.optimality),
        active_indices=np.flatnonzero(lower_active | upper_active).tolist(),
        objective=objective,
    )


def assess(p, y, angles, uv, reference, cfg, thresholds):
    mean, sd, lower, upper = prior_and_bounds(reference, cfg)
    prediction, jac = projection(p, angles, uv)
    residual = np.r_[(prediction - y) / cfg["noise_sigma"], (p - mean) / sd]
    augmented = np.vstack([jac / cfg["noise_sigma"], np.diag(1 / sd)])
    return assess_linearization(p, residual, augmented, sd, lower, upper, thresholds)
