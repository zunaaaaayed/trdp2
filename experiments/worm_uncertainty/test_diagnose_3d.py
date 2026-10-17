import unittest
import numpy as np
from diagnose_3d import features, voxel_features, ray_basis


class ThreeDTests(unittest.TestCase):
    def test_projection_parameter_jacobian(self):
        p = np.array([0.1, -0.2, 0.05, 0.7])
        s = np.array([[0.2, 0.3, 0.1]])
        angles = np.array([0.0, 0.7])
        uv = np.array([[0.1, 0.2], [-0.2, -0.1]])
        _, j, _ = features(p, s, angles, uv)
        for k in range(4):
            delta = np.zeros(4)
            delta[k] = 1e-6
            numeric = (
                features(p + delta, s, angles, uv)[0] - features(p - delta, s, angles, uv)[0]
            ) / 2e-6
            np.testing.assert_allclose(numeric, j[:, k], atol=1e-8, rtol=1e-5)

    def test_voxel_centre_jacobian(self):
        centres = np.array([[0.1, -0.2, 0.05]])
        s = np.array([[0.2, 0.3, 0.1]])
        points = np.array([[0.1, 0.2, 0.03], [-0.1, -0.1, 0]])
        _, derivative = voxel_features(centres, s, points)
        for k in range(3):
            delta = np.zeros_like(centres)
            delta[0, k] = 1e-6
            numeric = (
                voxel_features(centres + delta, s, points)[0]
                - voxel_features(centres - delta, s, points)[0]
            ) / 2e-6
            np.testing.assert_allclose(numeric[:, 0], derivative[:, 0, k], atol=1e-8)

    def test_exact_representable_density_solve(self):
        c = np.array([[0.1, -0.2, 0.05], [-0.3, 0.2, -0.1]])
        s = np.array([[0.2, 0.3, 0.1], [0.1, 0.2, 0.2]])
        uv = np.array([[0.1, 0.2], [-0.2, -0.1], [0.3, 0]])
        b, _ = ray_basis(c, s, np.array([0, 0.7, 1.5]), uv)
        density = np.array([0.8, 0.4])
        np.testing.assert_allclose(
            np.linalg.lstsq(b, b @ density, rcond=None)[0], density, atol=1e-12
        )
