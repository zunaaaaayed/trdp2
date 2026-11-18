"""Precision inversion, coupled variance and exact propagation identities."""

import unittest
import numpy as np
from experiments.worm_uncertainty.core.cholesky_gaussian import projection, voxel
from experiments.worm_uncertainty.core.posterior_local import (
    gaussian_factor,
    projection_mean,
    voxel_mean,
    interval_metrics,
)


class LocalPosteriorTests(unittest.TestCase):
    def test_covariance_factor_matches_direct_inverse(self):
        rng = np.random.default_rng(43)
        j = rng.normal(size=(20, 5))
        sd = np.array([0.1, 0.3, 2.0, 1.0, 0.8])
        sigma = 0.2
        f, _ = gaussian_factor(j, sd, sigma)
        expected = np.linalg.inv(j.T @ j / sigma**2 + np.diag(1 / sd**2))
        np.testing.assert_allclose(f @ f.T, expected, rtol=1e-12, atol=1e-14)
        conditional, _ = gaussian_factor(j[:, -2:], sd[-2:], sigma)
        self.assertGreaterEqual(
            np.linalg.eigvalsh((f @ f.T)[-2:, -2:] - conditional @ conditional.T).min(), -1e-12
        )

    def test_mean_only_matches_analytic_and_density_is_linear(self):
        p = np.r_[[0.1, -0.1, 0.02], np.log([0.15, 0.1, 0.12]), [0.01, -0.02, 0.01], 2.0]
        angles = np.array([0.1, 0.7, 1.3])
        uv = np.array([[u, z] for u in [-0.3, 0, 0.2] for z in [-0.1, 0.1]])
        points = np.array([[0.1, 0.2, 0.03], [-0.2, 0.1, -0.04], [0, 0, 0]])
        for analytic, fast, args in [
            (projection, projection_mean, (angles, uv)),
            (voxel, voxel_mean, (points,)),
        ]:
            baseline, j = analytic(p, *args)
            np.testing.assert_allclose(fast(p, *args), baseline, rtol=1e-12, atol=1e-14)
            for delta in [-3.0, 0.3]:
                q = p.copy()
                q[-1] += delta
                np.testing.assert_allclose(fast(q, *args), baseline + j[:, -1] * delta, atol=1e-13)

    def test_intervals_use_pointwise_truth_and_mask(self):
        samples = np.array([[0.0, 100.0], [1.0, 100.0], [2.0, 100.0], [3.0, 100.0]])
        metrics, _, _, _ = interval_metrics(samples, np.array([1.5, -1.0]), np.array([True, False]))
        self.assertEqual(metrics["coverage95"], 1.0)
        self.assertAlmostEqual(metrics["mean_rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
