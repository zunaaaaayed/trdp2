"""Finite acquired-direction probes preserve bounds and density linearity."""

import unittest
import numpy as np
from experiments.worm_uncertainty.studies.phantom_weak_directions import probes


class WeakDirectionTests(unittest.TestCase):
    def test_density_probes_are_linear_and_nonmutating(self):
        p = np.r_[[0.1, -0.1, 0.02], np.log([0.15, 0.1, 0.12]), np.zeros(3), 2.0]
        original = p.copy()
        cfg = dict(
            position_prior_sd=0.3,
            cholesky_log_diagonal_prior_sd=0.5,
            cholesky_off_diagonal_prior_sd=0.1,
            density_prior_sd=10.0,
            position_bounds=[-0.95, 0.95],
            density_bounds=[0.001, 100.0],
            noise_sigma=0.05,
        )
        angles = np.array([0.0, 0.5, 1.0, 1.5])
        uv = np.array([[u, z] for u in np.linspace(-0.5, 0.5, 10) for z in [-0.1, 0, 0.1]])
        rows = probes(p, angles, uv, np.array([True, True, True, False]), p, cfg)
        self.assertEqual(len(rows), 12)
        np.testing.assert_array_equal(p, original)
        for row in rows:
            self.assertFalse(row["skipped"])
            self.assertLessEqual(row["step"], row["requested_step"])
            self.assertGreaterEqual(row["density_relative_change"], 0)
            if row["mode"] == "density":
                self.assertLess(row["linearization_relative_error"], 1e-10)
                self.assertAlmostEqual(
                    row["density_relative_change"], row["step"] * 10 / 2, places=9
                )


if __name__ == "__main__":
    unittest.main()
