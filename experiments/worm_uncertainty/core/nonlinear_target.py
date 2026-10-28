"""Exact bounded target in original coordinates and stable importance diagnostics."""

import numpy as np
from scipy.special import logsumexp
from experiments.worm_uncertainty.core.posterior_local import projection_mean


class Target:
    """Only acquired observations enter this object. Constants independent of p omitted."""

    def __init__(self, y, angles, uv, sigma, mean, sd, lower, upper):
        if sigma <= 0 or np.any(sd <= 0):
            raise ValueError("positive noise/prior scales required")
        self.y, self.angles, self.uv = y, angles, uv
        self.sigma, self.mean, self.sd = sigma, mean, sd
        self.lower, self.upper = lower, upper

    def evaluate(self, p):
        valid = bool(np.isfinite(p).all() and np.all(p >= self.lower) and np.all(p <= self.upper))
        pred = projection_mean(p, self.angles, self.uv)
        r = (pred - self.y) / self.sigma
        prior = (p - self.mean) / self.sd
        ll, lp = -0.5 * float(r @ r), -0.5 * float(prior @ prior)
        return valid, ll, lp, ll + lp if valid else -np.inf, pred

    def logpdf(self, p):
        if not np.isfinite(p).all() or np.any(p < self.lower) or np.any(p > self.upper):
            return -np.inf
        return self.evaluate(p)[3]


def proposal_logpdf(draws, mean, factor):
    """Square nonsingular factor, in the SAME active coordinates as target measure."""
    z = np.linalg.solve(factor, (draws - mean).T).T
    sign, logdet = np.linalg.slogdet(factor)
    if sign == 0:
        raise ValueError("singular proposal")
    return -0.5 * np.sum(z * z, axis=1) - logdet - len(mean) * np.log(2 * np.pi) / 2


def weights(logweights):
    logweights = np.asarray(logweights)
    if np.isnan(logweights).any() or np.isposinf(logweights).any():
        raise ValueError("invalid log weights")
    if not np.isfinite(logweights).any():
        return np.zeros(len(logweights)), dict(ess=0.0, ess_fraction=0.0, maximum_weight=1.0)
    w = np.exp(logweights - logsumexp(logweights))
    ess = float(1 / (w @ w))
    return w, dict(ess=ess, ess_fraction=ess / len(w), maximum_weight=float(w.max()))


def weighted_summary(values, truth, mask, w):
    """Weighted inverse-CDF quantiles; proper interval score, not a calibration test."""
    x, t = values[:, mask], truth[mask]
    if w.sum() <= 0 or not np.isfinite(x).all():
        raise ValueError("finite outputs and nonzero weights required")
    w = w / w.sum()
    order = np.argsort(x, axis=0)
    sorted_x = np.take_along_axis(x, order, axis=0)
    cumulative = np.cumsum(w[order], axis=0)
    lo, hi = [
        sorted_x[(cumulative >= q).argmax(axis=0), np.arange(x.shape[1])] for q in [0.025, 0.975]
    ]
    mean = w @ x
    width = hi - lo
    score = width + 40 * np.maximum(lo - t, 0) + 40 * np.maximum(t - hi, 0)
    return dict(
        coverage=float(np.mean((t >= lo) & (t <= hi))),
        width=float(width.mean()),
        relative_l2=float(np.linalg.norm(mean - t) / max(np.linalg.norm(t), 1e-30)),
        interval_score=float(score.mean()),
        mean=float(mean.mean()),
    )


def split_diagnostics(chains):
    """Classical split Rhat and conservative positive-autocorrelation ESS; feasibility only.

    Not rank-normalized diagnostics; a pass is necessary, never a reference certificate.
    """
    chains = np.asarray(chains)
    m, n, d = chains.shape
    halves = np.concatenate([chains[:, : n // 2], chains[:, -n // 2 :]], axis=0)
    m, n, d = halves.shape
    within = np.mean(np.var(halves, axis=1, ddof=1), axis=0)
    between = n * np.var(np.mean(halves, axis=1), axis=0, ddof=1)
    var = (n - 1) / n * within + between / n
    rhat = np.sqrt(var / np.maximum(within, 1e-300))
    centred = halves - halves.mean(axis=1, keepdims=True)
    fft = np.fft.rfft(centred, n=2 * n, axis=1)
    acov = np.fft.irfft(fft.conj() * fft, axis=1)[:, :n].real / n
    rho = 1 - (within[None, :] - acov.mean(axis=0)) / np.maximum(var, 1e-300)
    rho[0] = 1
    pairs = rho[0:-1:2] + rho[1::2]
    pairs = np.minimum.accumulate(pairs, axis=0)
    pairs = np.maximum(pairs, 0)
    tau = np.maximum(1, -1 + 2 * pairs.sum(axis=0))
    ess = np.minimum(m * n, m * n / tau)
    return dict(
        maximum_split_rhat=float(rhat.max()),
        minimum_ess=float(ess.min()),
        split_rhat=rhat.tolist(),
        ess=ess.tolist(),
    )


def metropolis(target, mean, factor, start, steps, step, seed):
    """Fixed symmetric affine random walk, exact hard-bound rejection; no clipping."""
    rng = np.random.default_rng(seed)
    p, logp = start.copy(), target.logpdf(start)
    if not np.isfinite(logp):
        raise ValueError("invalid initial state")
    draws, logs = np.empty((steps, len(p))), np.empty(steps)
    accepted = 0
    for i in range(steps):
        candidate = p + step * (factor @ rng.normal(size=len(p)))
        value = target.logpdf(candidate)
        if np.log(rng.uniform()) < value - logp:
            p, logp = candidate, value
            accepted += 1
        draws[i], logs[i] = p, logp
    return draws, logs, accepted / steps
