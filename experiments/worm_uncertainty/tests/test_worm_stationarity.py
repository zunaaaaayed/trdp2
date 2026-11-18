"""Check stationarity gradient against finite differences and continuation semantics."""

import unittest
import numpy as np
from experiments.worm_uncertainty.core.cholesky_gaussian import projection
from experiments.worm_uncertainty.studies.worm_stationarity import diagnostics, continue_fit
from experiments.worm_uncertainty.core.staged_density import prior_and_bounds


class StationarityTests(unittest.TestCase):
    def setUp(self):
        self.p = np.r_[[0.1, -0.1, 0.02], np.log([0.15, 0.1, 0.12]), np.zeros(3), 2.0]
        self.cfg = dict(
            position_prior_sd=0.3,
            cholesky_log_diagonal_prior_sd=0.5,
            cholesky_off_diagonal_prior_sd=0.1,
            density_prior_sd=10.0,
            position_bounds=[-0.95, 0.95],
            density_bounds=[0.001, 100.0],
            noise_sigma=0.05,
        )
        self.angles = np.array([0.0, 0.5, 1.0])
        self.uv = np.array([[u, z] for u in np.linspace(-0.5, 0.5, 10) for z in [-0.1, 0, 0.1]])
        self.y = projection(self.p, self.angles, self.uv)[0]

    def test_gradient_and_residual_statistics(self):
        p = self.p.copy()
        p[-1] = 1.8
        d = diagnostics(p, self.y, self.angles, self.uv, self.p, self.cfg)
        mean, sd, low, high = prior_and_bounds(self.p, self.cfg)

        def cost(x):
            r = (projection(x, self.angles, self.uv)[0] - self.y) / 0.05
            return 0.5 * (r @ r + np.sum(((x - mean) / sd) ** 2))

        g = np.empty(len(p))
        for i in range(len(p)):
            h = np.zeros(len(p))
            h[i] = 1e-6
            g[i] = (cost(p + h) - cost(p - h)) / 2e-6
        np.testing.assert_allclose(
            d["projected_gradient_inf"], np.max(abs(p - np.clip(p - g, low, high))), rtol=1e-6
        )
        np.testing.assert_allclose(d["raw_gradient_inf"], np.max(abs(g)), rtol=1e-6)
        self.assertAlmostEqual(d["objective"], 2 * cost(p), places=9)
        self.assertTrue(np.isfinite(d["principal_widths"]).all())

    def test_budget_exhaustion_not_convergence(self):
        p = self.p.copy()
        p[-1] = 1.0
        design = dict(additional_nfev=1, xtol=1e-14, gtol=1e-6, projected_gradient_tolerance=1e-6)
        f = continue_fit(p, self.y, self.angles, self.uv, self.p, self.cfg, design)
        self.assertFalse(f["converged"])
        self.assertEqual(f["status"], 0)
        self.assertLessEqual(f["after"]["objective"], f["before"]["objective"])


if __name__ == "__main__":
    unittest.main()
