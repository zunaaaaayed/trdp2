"""Geometry tie ordering must not create apparent risk-ranking skill."""

import unittest
import numpy as np
from experiments.worm_uncertainty.studies.diversity_risk_sensitivity import tied_risk_curve


class RiskTieTests(unittest.TestCase):
    def test_constant_score_has_flat_expected_risk(self):
        e = np.array([1.0, 2.0, 100.0, 200.0])
        s = np.ones(4)
        np.testing.assert_allclose(tied_risk_curve(e, s, [0.25, 0.5, 1]), [75.75] * 3)
        np.testing.assert_allclose(tied_risk_curve(e[::-1], s, [0.25, 0.5, 1]), [75.75] * 3)

    def test_partial_tie_and_perfect_order(self):
        self.assertEqual(tied_risk_curve([1, 3, 5, 9], [0, 1, 1, 2], [0.5]), [2.5])
        self.assertEqual(tied_risk_curve([1, 2, 3, 4], [1, 2, 3, 4], [0.5, 1]), [1.5, 2.5])


if __name__ == "__main__":
    unittest.main()
