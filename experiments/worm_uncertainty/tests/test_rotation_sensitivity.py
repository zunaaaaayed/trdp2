"""Ensure the diagnostic control reproduces the original objective and solver."""

import unittest
import numpy as np
from experiments.worm_uncertainty.core.shape_gaussian import fit_shape, projection
from experiments.worm_uncertainty.studies.rotation_sensitivity import fit_control


class RotationControlTests(unittest.TestCase):
    def test_original_bounded_fit_is_reproduced(self):
        p = np.r_[[0.03, -0.02, 0.01], np.log([0.17, 0.11, 0.2]), [0.1, -0.2, 0.3], 0.7]
        angles = np.array([0, 0.5, 1.0, 1.5])
        uv = np.array([[u, v] for u in [-0.3, 0, 0.3] for v in [-0.2, 0, 0.2]])
        y = projection(p, angles, uv)[0]
        cfg = dict(
            position_prior_sd=0.3,
            log_width_prior_sd=0.35,
            rotation_prior_sd=0.5,
            density_prior_sd=3,
            position_bounds=[-0.75, 0.75],
            width_bounds=[0.04, 0.35],
            rotation_bound=1.4,
            density_bounds=[0.001, 3],
            budget_nfev=150,
        )
        initial = p.copy()
        initial[-1] = 0.5
        original, sd = fit_shape(y, angles, uv, initial, 0.01, cfg)
        control, control_sd = fit_control(y, angles, uv, initial, initial, 0.01, cfg, True)
        np.testing.assert_array_equal(original.x, control.x)
        np.testing.assert_array_equal(sd, control_sd)
        self.assertEqual(original.cost, control.cost)


if __name__ == "__main__":
    unittest.main()
