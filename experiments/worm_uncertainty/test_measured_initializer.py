import unittest
import json
from pathlib import Path
import numpy as np
from measured_initializer import initialize, perturb
from diagnose_3d import ray_basis


class MeasuredInitializerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(
            Path(__file__).with_name("measured_init_protocol_v1.json").read_text()
        )
        self.cfg.update(components=1, initialization_grid_xy=5, initialization_grid_z=3)
        self.angles = np.array([0.0, 0.5, 1.0])
        self.uv = np.array(
            [[u, v] for u in np.linspace(-0.8, 0.8, 9) for v in np.linspace(-0.8, 0.8, 7)]
        )

    def test_recover_single_dictionary_atom_and_reproducible_starts(self):
        centre = np.array([[0.25, -0.25, 0.0]])
        y = ray_basis(centre, np.full((1, 3), 0.14), self.angles, self.uv)[0].ravel() * 0.7
        reference, info = initialize(y, self.angles, self.uv, self.cfg)
        np.testing.assert_allclose(reference[:3], centre.ravel(), atol=1e-14)
        self.assertAlmostEqual(reference[-1], 0.7, places=12)
        self.assertLess(info["acquired_relative_residual"], 1e-12)
        np.testing.assert_array_equal(
            perturb(reference, 81, self.cfg), perturb(reference, 81, self.cfg)
        )
        np.testing.assert_array_equal(perturb(reference, 0, self.cfg), reference)

    def test_withheld_changes_do_not_change_reference(self):
        angles = np.r_[self.angles, [1.5, 2.0, 2.5]]
        y = (
            ray_basis(np.array([[0.25, -0.25, 0.0]]), np.full((1, 3), 0.14), angles, self.uv)[
                0
            ].ravel()
            * 0.7
        )
        acquired = np.repeat([True, True, True, False, False, False], len(self.uv))
        before = initialize(y[acquired], angles[:3], self.uv, self.cfg)[0]
        y[~acquired] = 1e9
        after = initialize(y[acquired], angles[:3], self.uv, self.cfg)[0]
        np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
