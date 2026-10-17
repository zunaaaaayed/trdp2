import unittest
import numpy as np
from bayes_linear import posterior


class PosteriorTests(unittest.TestCase):
    def test_scalar_closed_form(self):
        gain, cov = posterior(np.array([[2.0]]), 0.5, 3.0)
        expected = 1 / (4 / 0.25 + 1 / 9)
        self.assertAlmostEqual(cov[0, 0], expected)
        self.assertAlmostEqual(gain[0, 0], expected * 2 / 0.25)

    def test_unmeasured_direction_retains_prior(self):
        _, cov = posterior(np.array([[1.0, 0.0]]), 0.1, 2.0)
        self.assertAlmostEqual(cov[1, 1], 4.0)
        self.assertLess(cov[0, 0], 0.01)

    def test_removed_data_increases_covariance(self):
        a = np.random.default_rng(1).normal(size=(8, 4))
        _, full = posterior(a, 0.2, 1.0)
        _, partial = posterior(a[:4], 0.2, 1.0)
        self.assertGreaterEqual(np.linalg.eigvalsh(partial - full).min(), -1e-12)
