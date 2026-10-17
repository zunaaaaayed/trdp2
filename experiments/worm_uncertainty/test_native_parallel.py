import unittest
import numpy as np
from native_parallel import camera_basis, analytic_projection
from fit_gaussian_slab import slab_layout


class ParallelTests(unittest.TestCase):
    def test_basis_and_detector_directions(self):
        for angle in [0, 0.7, np.pi / 2]:
            basis = camera_basis(angle)
            np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-6)
            self.assertAlmostEqual(np.linalg.det(basis), 1, places=6)
            np.testing.assert_allclose(basis[:, 1], [0, 0, 1])

    def test_isotropic_rotation_invariance(self):
        a = analytic_projection([[0, 0, 0]], [[0.2, 0.2, 0.2]], [1], 0, 32)
        b = analytic_projection([[0, 0, 0]], [[0.2, 0.2, 0.2]], [1], 0.7, 32)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_slab_spacing_and_centres(self):
        rows, spacing = slab_layout(8)
        np.testing.assert_allclose(
            (np.arange(rows.start, rows.stop) + 0.5) * spacing - 1,
            (np.arange(8) + 0.5 - 4) * spacing,
            atol=1e-15,
        )
        with self.assertRaises(ValueError):
            slab_layout(7)
