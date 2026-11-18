"""Checks independent of research outcomes."""

import unittest
import numpy as np
from scipy.stats import multivariate_normal
from experiments.worm_uncertainty.core.nonlinear_target import (
    proposal_logpdf,
    weights,
    weighted_summary,
    split_diagnostics,
    metropolis,
    Target,
)
from experiments.worm_uncertainty.core.posterior_local import projection_mean


class TestTarget(unittest.TestCase):
    def test_proposal_general_factor(self):
        f = np.array([[2.0, 1.0], [0.3, 0.9]])
        x = np.array([[1.0, 2.0], [-1.0, 0.5]])
        np.testing.assert_allclose(
            proposal_logpdf(x, np.zeros(2), f), multivariate_normal.logpdf(x, cov=f @ f.T)
        )

    def test_logweights(self):
        w, d = weights(np.array([-100000.0, -100000.0, -np.inf]))
        np.testing.assert_allclose(w, [0.5, 0.5, 0])
        self.assertAlmostEqual(d["ess"], 2)
        self.assertEqual(weights([-np.inf, -np.inf])[1]["ess"], 0)
        with self.assertRaises(ValueError):
            weights([np.nan])

    def test_target_factor_bounds(self):
        p = np.r_[0.0, 0.0, 0.0, np.log([0.1, 0.2, 0.3]), 0.0, 0.0, 0.0, 1.0]
        angles = np.array([0.0, 0.5])
        uv = np.array([[0.0, 0.0], [0.2, 0.1]])
        y = projection_mean(p, angles, uv) + 0.1
        lower = np.full(10, -np.inf)
        upper = np.full(10, np.inf)
        lower[-1] = 0.001
        t = Target(y, angles, uv, 0.2, np.zeros(10), np.ones(10), lower, upper)
        self.assertAlmostEqual(t.logpdf(p), -0.5 * (1 + p @ p))
        bad = p.copy()
        bad[-1] = -1
        self.assertEqual(t.logpdf(bad), -np.inf)
        # No withheld values or truth arguments exist on Target.
        self.assertEqual(t.y.shape, (4,))

    def test_linear_gaussian_importance(self):
        rng = np.random.default_rng(3)
        x = rng.normal(2, 0.5, (1000, 1))
        target = -0.5 * ((x[:, 0] - 2) / 0.5) ** 2
        w, d = weights(target - proposal_logpdf(x, np.array([2.0]), np.array([[0.5]])))
        np.testing.assert_allclose(w, np.full(1000, 0.001))
        self.assertAlmostEqual(d["ess"], 1000)

    def test_score_and_diagnostics(self):
        x = np.arange(1000.0)[:, None] / 1000
        r = weighted_summary(x, np.array([0.5]), np.array([True]), np.ones(1000))
        self.assertEqual(r["coverage"], 1)
        self.assertAlmostEqual(r["interval_score"], r["width"])
        rng = np.random.default_rng(19)
        c = rng.normal(size=(4, 1000, 2))
        self.assertLess(split_diagnostics(c)["maximum_split_rhat"], 1.01)
        c[0, :, 0] += 4
        self.assertGreater(split_diagnostics(c)["maximum_split_rhat"], 1.1)

    def test_bounded_scalar_sampler(self):
        class Scalar:
            def logpdf(self, p):
                return -0.5 * p[0] ** 2 if p[0] >= 0 else -np.inf

        x, _, _ = metropolis(Scalar(), np.zeros(1), np.eye(1), np.ones(1), 12000, 1.0, 9)
        self.assertTrue((x >= 0).all())
        self.assertAlmostEqual(x[1000:].mean(), np.sqrt(2 / np.pi), delta=0.06)


if __name__ == "__main__":
    unittest.main()
