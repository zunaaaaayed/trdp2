import unittest
import numpy as np
from experiments.worm_uncertainty.core.langevin_reference import (
    mala,
    rank_diagnostics,
    AffineTarget,
)
from experiments.worm_uncertainty.core.nonlinear_target import Target
from experiments.worm_uncertainty.core.posterior_local import projection_mean


class LangevinTests(unittest.TestCase):
    def test_linear_gaussian_control(self):
        precision = np.array([[2.0, 0.5], [0.5, 1.0]])

        def value(x):
            return -0.5 * x @ precision @ x, -precision @ x

        x, _, _ = mala(value, np.zeros(2), 0.6, 18000, 82)
        np.testing.assert_allclose(x[2000:].mean(axis=0), 0, atol=0.06)
        np.testing.assert_allclose(
            np.cov(x[2000:].T), np.linalg.inv(precision), rtol=0.1, atol=0.03
        )

    def test_bounded_scalar(self):
        def value(x):
            return (-0.5 * x[0] ** 2, -x) if x[0] >= 0 else (-np.inf, np.zeros(1))

        x, _, _ = mala(value, np.ones(1), 0.8, 14000, 83)
        self.assertTrue((x >= 0).all())
        self.assertAlmostEqual(x[1000:].mean(), np.sqrt(2 / np.pi), delta=0.05)

    def test_rank_diagnostics(self):
        rng = np.random.default_rng(74)
        x = rng.normal(size=(4, 1000, 2))
        self.assertTrue(rank_diagnostics(x)["passed"])
        x[0, :, 0] *= 4
        self.assertFalse(rank_diagnostics(x)["passed"])
        self.assertFalse(rank_diagnostics(np.ones((4, 100, 1)))["passed"])

    def test_affine_gradient(self):
        p = np.r_[0.1, 0.0, 0.0, np.log([0.1, 0.2, 0.3]), 0.0, 0.0, 0.0, 1.0]
        a = np.array([0.0, 0.8])
        uv = np.array([[0.0, 0.0], [0.1, 0.2]])
        t = Target(
            projection_mean(p, a, uv) + 0.01,
            a,
            uv,
            0.1,
            np.zeros(10),
            np.ones(10),
            np.full(10, -10.0),
            np.full(10, 10.0),
        )
        target = AffineTarget(t, p, np.eye(10) * 0.1)
        q = np.zeros(10)
        v, g = target.value_gradient(q)
        self.assertAlmostEqual(v, t.logpdf(p))
        for i in range(10):
            h = np.eye(10)[i] * 1e-5
            fd = (target.value_gradient(q + h)[0] - target.value_gradient(q - h)[0]) / 2e-5
            self.assertAlmostEqual(g[i], fd, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
