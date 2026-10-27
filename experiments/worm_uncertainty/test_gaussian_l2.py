"""Analytic integral identities, translation and component permutation invariance."""

import unittest
import numpy as np
from gaussian_l2 import inner_product, relative_l2


class GaussianL2Tests(unittest.TestCase):
    def test_single_gaussian_integral_and_amplitude(self):
        p = np.r_[[0.0, 0.0, 0.0], np.log([0.1, 0.2, 0.3]), np.zeros(3), 2.0]
        self.assertAlmostEqual(inner_product(p, p), 4 * np.pi**1.5 * 0.1 * 0.2 * 0.3, places=12)
        self.assertEqual(relative_l2(p, p), 0.0)
        q = p.copy()
        q[-1] *= 1.2
        self.assertAlmostEqual(relative_l2(q, p), 0.2, places=12)
        q = p.copy()
        q[0] = 0.1
        self.assertAlmostEqual(relative_l2(q, p), np.sqrt(2 * (1 - np.exp(-0.25))), places=12)


if __name__ == "__main__":
    unittest.main()
