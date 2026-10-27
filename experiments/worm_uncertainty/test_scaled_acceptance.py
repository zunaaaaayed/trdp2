"""Unit invariance, feasible KKT signs, positive and unfinished controls."""

import unittest
import numpy as np
from scaled_acceptance import assess_linearization


class ScaledAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.thresholds = dict(
            bound_tolerance=1e-8,
            normalized_gradient=1e-6,
            relative_predicted_decrease=1e-8,
            prior_scaled_correction_inf=1e-3,
        )

    def test_parameter_unit_invariance(self):
        p = np.array([0.2, 0.3])
        sd = np.array([0.1, 5.0])
        j = np.array([[1.0, 2.0], [3.0, -0.5], [0.7, 1.3]])
        r = np.array([0.2, -0.1, 0.3])
        lo = np.array([-1.0, -2.0])
        hi = np.array([2.0, 3.0])
        a = assess_linearization(p, r, j, sd, lo, hi, self.thresholds)
        scale = np.array([1000.0, 0.001])
        b = assess_linearization(
            p * scale, r, j / scale, sd * scale, lo * scale, hi * scale, self.thresholds
        )
        for key in [
            "normalized_gradient",
            "relative_predicted_decrease",
            "prior_scaled_correction_inf",
        ]:
            self.assertAlmostEqual(a[key], b[key], places=11)
        self.assertEqual(a["accepted"], b["accepted"])

    def test_accepts_exact_optimum_rejects_unfinished_and_respects_bounds(self):
        p = np.array([0.0, 0.0])
        j = np.eye(2)
        sd = np.ones(2)
        lo = -np.ones(2)
        hi = np.ones(2)
        self.assertTrue(
            assess_linearization(p, np.zeros(2), j, sd, lo, hi, self.thresholds)["accepted"]
        )
        self.assertFalse(
            assess_linearization(p, np.ones(2), j, sd, lo, hi, self.thresholds)["accepted"]
        )
        at_bound = assess_linearization(lo, np.ones(2), j, sd, lo, hi, self.thresholds)
        self.assertTrue(at_bound["accepted"])
        self.assertEqual(at_bound["active_indices"], [0, 1])

    def test_tiny_residual_does_not_hide_large_weak_direction_correction(self):
        p = np.zeros(2)
        r = np.array([0.0, 1e-9])
        j = np.diag([1.0, 1e-8])
        sd = np.ones(2)
        result = assess_linearization(p, r, j, sd, -np.ones(2), np.ones(2), self.thresholds)
        self.assertFalse(result["accepted"])
        self.assertGreater(result["prior_scaled_correction_inf"], 0.01)


if __name__ == "__main__":
    unittest.main()
