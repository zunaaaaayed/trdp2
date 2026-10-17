import unittest
import numpy as np
import shape_gaussian
from cholesky_gaussian import convert_euler, projection, voxel, geometry


class CholeskyTests(unittest.TestCase):
    def test_same_physical_covariance_and_features(self):
        p = np.r_[[0.1, -0.2, 0.03], np.log([0.17, 0.11, 0.2]), [0.2, -0.3, 0.4], 0.7]
        converted = convert_euler(p)
        np.testing.assert_allclose(
            geometry(converted)[1], shape_gaussian.geometry(p)[1], atol=1e-15
        )
        angles = np.array([0, 0.7, 1.5])
        uv = np.array([[0.1, 0.03], [-0.2, 0.1]])
        np.testing.assert_allclose(
            projection(converted, angles, uv)[0],
            shape_gaussian.projection(p, angles, uv)[0],
            atol=1e-14,
        )
        points = np.c_[uv, np.zeros(2)]
        np.testing.assert_allclose(
            voxel(converted, points)[0], shape_gaussian.voxel(p, points)[0], atol=1e-14
        )
        for function in [lambda x: projection(x, angles, uv), lambda x: voxel(x, points)]:
            _, jac = function(converted)
            for j in range(len(converted)):
                delta = np.zeros_like(converted)
                delta[j] = 1e-6
                numeric = (function(converted + delta)[0] - function(converted - delta)[0]) / 2e-6
                np.testing.assert_allclose(jac[:, j], numeric, atol=2e-8, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
