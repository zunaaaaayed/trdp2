import unittest
import numpy as np
from scipy.integrate import quad
from shape_gaussian import projection, voxel, geometry, uncertainty
from diagnose_3d import ray_basis


class ShapeTests(unittest.TestCase):
    def setUp(self):
        self.p = np.r_[[0.1, -0.15, 0.03], np.log([0.17, 0.11, 0.23]), [0.2, -0.3, 0.35], 0.7]
        self.angles = np.array([0, 0.6, 1.7])
        self.uv = np.array([[0.1, 0.05], [-0.2, -0.1]])

    def test_all_projection_and_voxel_derivatives(self):
        for function in [
            lambda p: projection(p, self.angles, self.uv),
            lambda p: voxel(p, np.c_[self.uv, np.zeros(2)]),
        ]:
            _, jac = function(self.p)
            for j in range(len(self.p)):
                delta = np.zeros_like(self.p)
                delta[j] = 1e-6
                numeric = (function(self.p + delta)[0] - function(self.p - delta)[0]) / 2e-6
                np.testing.assert_allclose(jac[:, j], numeric, rtol=2e-5, atol=2e-9)

    def test_multiple_component_parameter_order(self):
        p = np.r_[
            self.p[:3],
            [-0.2, 0.1, -0.04],
            self.p[3:6],
            np.log([0.1, 0.21, 0.14]),
            self.p[6:9],
            [-0.1, 0.2, -0.25],
            [0.7, 0.4],
        ]
        for function in [
            lambda p: projection(p, self.angles, self.uv),
            lambda p: voxel(p, np.c_[self.uv, np.zeros(2)]),
        ]:
            _, jac = function(p)
            for j in range(len(p)):
                delta = np.zeros_like(p)
                delta[j] = 1e-6
                numeric = (function(p + delta)[0] - function(p - delta)[0]) / 2e-6
                np.testing.assert_allclose(jac[:, j], numeric, rtol=2e-5, atol=2e-9)

    def test_rotated_projection_against_ray_quadrature(self):
        c, s, _, d = geometry(self.p)
        inv = np.linalg.inv(s[0])
        actual = projection(self.p, self.angles, self.uv)[0].reshape(3, 2)
        for a, theta in enumerate(self.angles):
            beam = np.array([np.sin(theta), np.cos(theta), 0])
            uaxis = np.array([np.cos(theta), -np.sin(theta), 0])
            for i, (u, v) in enumerate(self.uv):
                origin = u * uaxis + np.array([0, 0, v])

                def integrand(t):
                    delta = origin + t * beam - c[0]
                    return d[0] * np.exp(-0.5 * delta @ inv @ delta)

                expected = quad(integrand, -3, 3, epsabs=1e-11)[0]
                self.assertAlmostEqual(actual[a, i], expected, places=10)

    def test_diagonal_reduction_and_nested_variances(self):
        p = self.p.copy()
        p[6:9] = 0
        pred, jac = projection(p, self.angles, self.uv)
        expected = (
            ray_basis(p[:3].reshape(1, 3), np.exp(p[3:6]).reshape(1, 3), self.angles, self.uv)[0]
            * p[-1]
        )
        np.testing.assert_allclose(pred, expected.ravel(), atol=1e-14)
        _, vj = voxel(p, np.array([[0, 0, 0], [0.1, 0.1, 0.1]]))
        result = uncertainty(jac, vj, np.ones(len(pred), bool), 0.01, np.ones(10), 1)
        for smaller, larger in zip(
            ["density", "centres", "widths"], ["centres", "widths", "rotations"]
        ):
            for i in [0, 1]:
                self.assertTrue(np.all(result[larger][i] ** 2 >= result[smaller][i] ** 2 - 1e-12))


if __name__ == "__main__":
    unittest.main()
