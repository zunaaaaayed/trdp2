"""Fixed-metric Metropolis-adjusted Langevin feasibility; exact hard-bound target."""

import time
import numpy as np
from scipy.stats import rankdata, norm
from experiments.worm_uncertainty.core.cholesky_gaussian import projection
from experiments.worm_uncertainty.core.nonlinear_target import split_diagnostics


class AffineTarget:
    """p=centre+factor@q; constant determinant cancels, original prior unchanged."""

    def __init__(self, target, centre, factor):
        self.target, self.centre, self.factor = target, centre, factor

    def value_gradient(self, q):
        p = self.centre + self.factor @ q
        t = self.target
        if not np.isfinite(p).all() or np.any(p < t.lower) or np.any(p > t.upper):
            return -np.inf, np.zeros_like(q)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                pred, jac = projection(p, t.angles, t.uv)
                r = (pred - t.y) / t.sigma
                prior = (p - t.mean) / t.sd
                value = -0.5 * (r @ r + prior @ prior)
                g = self.factor.T @ (-jac.T @ r / t.sigma - prior / t.sd)
                if not np.isfinite(value) or not np.isfinite(g).all():
                    return -np.inf, np.zeros_like(q)
                return float(value), g
            except (FloatingPointError, np.linalg.LinAlgError):
                return -np.inf, np.zeros_like(q)


def mala(value_gradient, start, step, steps, seed, deadline=None):
    rng = np.random.default_rng(seed)
    q = start.copy()
    value, g = value_gradient(q)
    if not np.isfinite(value):
        raise ValueError("invalid start")
    chain = []
    logs = []
    accepted = []
    jumps = []
    invalid = 0
    for _ in range(steps):
        if deadline is not None and time.monotonic() >= deadline:
            break
        forward = q + 0.5 * step**2 * g
        proposal = forward + step * rng.normal(size=len(q))
        newvalue, newg = value_gradient(proposal)
        accept = False
        jump = 0.0
        if np.isfinite(newvalue):
            reverse = proposal + 0.5 * step**2 * newg
            logratio = (
                newvalue
                - value
                - (np.sum((q - reverse) ** 2) - np.sum((proposal - forward) ** 2)) / (2 * step**2)
            )
            accept = np.log(rng.uniform()) < logratio
            if accept:
                jump = float(np.sum((proposal - q) ** 2))
                q, value, g = proposal, newvalue, newg
        else:
            invalid += 1
        chain.append(q.copy())
        logs.append(value)
        accepted.append(accept)
        jumps.append(jump)
    return (
        np.array(chain),
        np.array(logs),
        dict(
            acceptance=float(np.mean(accepted)) if accepted else 0.0,
            mean_squared_jump=float(np.mean(jumps)) if jumps else 0.0,
            invalid_proposals=invalid,
            completed_steps=len(chain),
        ),
    )


def rank_normalize(x):
    # Blom normal scores with average ranks for repeated MCMC states.
    return norm.ppf((rankdata(x, method="average").reshape(x.shape) - 0.375) / (x.size + 0.25))


def rank_diagnostics(chains):
    """Rank/folded split Rhat, bulk ESS and min 5/95% indicator ESS per variable."""
    rows = []
    for j in range(chains.shape[2]):
        x = chains[:, :, j]
        if np.ptp(x) == 0:
            rows.append(
                dict(rank_rhat=None, folded_rhat=None, bulk_ess=0.0, tail_ess=0.0, constant=True)
            )
            continue
        ranked = split_diagnostics(rank_normalize(x)[:, :, None])
        folded = split_diagnostics(rank_normalize(abs(x - np.median(x)))[:, :, None])
        tail = []
        for probability in [0.05, 0.95]:
            indicator = (x <= np.quantile(x, probability)).astype(float)
            tail.append(
                split_diagnostics(indicator[:, :, None])["minimum_ess"]
                if np.ptp(indicator) > 0
                else 0.0
            )
        rows.append(
            dict(
                rank_rhat=ranked["maximum_split_rhat"],
                folded_rhat=folded["maximum_split_rhat"],
                bulk_ess=ranked["minimum_ess"],
                tail_ess=min(tail),
                constant=False,
            )
        )
    rhat = max(
        [max(r["rank_rhat"], r["folded_rhat"]) for r in rows if not r["constant"]], default=None
    )
    return dict(
        variables=rows,
        constant_variables=sum(r["constant"] for r in rows),
        maximum_rank_folded_rhat=rhat,
        minimum_bulk_ess=min(r["bulk_ess"] for r in rows),
        minimum_tail_ess=min(r["tail_ess"] for r in rows),
        passed=bool(
            not any(r["constant"] for r in rows)
            and rhat <= 1.01
            and min(r["bulk_ess"] for r in rows) >= 400
            and min(r["tail_ess"] for r in rows) >= 400
        ),
    )
