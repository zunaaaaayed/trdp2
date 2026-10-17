"""Acquisition wrapping and held-out data isolation for the stress study."""

import unittest
import numpy as np
from stress_3d import acquisition_mask, evaluate
from diagnose_3d import features


class StressTests(unittest.TestCase):
    def test_wrapped_window(self):
        degrees = np.arange(0, 180, 5)
        mask = acquisition_mask(degrees, 120, 90)
        np.testing.assert_array_equal(
            degrees[mask], np.r_[np.arange(0, 30, 5), np.arange(120, 180, 5)]
        )
        self.assertEqual(mask.sum(), 18)
        self.assertTrue(acquisition_mask(degrees, 60, 180).all())

    def test_withheld_values_do_not_change_fit_or_covariance(self):
        angles = np.deg2rad(np.arange(0, 180, 30))
        uv = np.array([[x, z] for x in [-0.4, 0, 0.4] for z in [-0.2, 0, 0.2]])
        scales = np.array([[0.2, 0.15, 0.2]])
        params = np.array([0.03, -0.04, 0.02, 0.7])
        y = features(params, scales, angles, uv)[0]
        mask = np.array([True, True, True, False, False, False])
        altered = y.copy()
        altered[~np.repeat(mask, len(uv))] += 1e6
        cfg = dict(
            position_prior_sd=0.3,
            density_prior_sd=3,
            position_bounds=[-0.75, 0.75],
            density_bounds=[0.001, 3],
            budget_nfev=150,
        )
        a = evaluate(y, angles, uv, scales, params, 0.01, cfg, mask, uv[:, [0, 0, 1]])
        b = evaluate(altered, angles, uv, scales, params, 0.01, cfg, mask, uv[:, [0, 0, 1]])
        for key in [
            "ray_mean",
            "voxel_mean",
            "density_ray_sd",
            "joint_ray_sd",
            "density_voxel_sd",
            "joint_voxel_sd",
        ]:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertTrue(np.all(a["joint_voxel_sd"] ** 2 >= a["density_voxel_sd"] ** 2 - 1e-12))


if __name__ == "__main__":
    unittest.main()
