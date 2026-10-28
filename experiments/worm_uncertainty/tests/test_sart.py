import unittest
import numpy as np
from experiments.worm_uncertainty.studies.compare_sart import fit_sart


class SartTests(unittest.TestCase):
    def test_withheld_isolation_and_nonnegative(self):
        angles = np.arange(0, 180, 18.0)
        a = np.random.default_rng(5).normal(size=(10, 16))
        mask = angles < 90
        b = a.copy()
        b[~mask] = 1e9
        x = fit_sart(a, angles, mask, iterations=2)
        np.testing.assert_array_equal(x, fit_sart(b, angles, mask, iterations=2))
        self.assertTrue(np.isfinite(x).all() and (x >= 0).all())
