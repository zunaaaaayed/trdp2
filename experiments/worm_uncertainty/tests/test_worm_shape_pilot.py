from experiments.worm_uncertainty.layout import resource_path
import json
import unittest
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.studies.worm_shape_pilot import (
    sampling_geometry,
    normalize_acquired,
    initialize_worm,
)
from experiments.worm_uncertainty.studies.diagnose_3d import ray_basis
from experiments.worm_uncertainty.studies.stress_3d import acquisition_mask


class WormShapeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(resource_path("worm_shape_protocol_v1.json").read_text())

    def test_sampled_geometry_and_measured_normalization(self):
        angles, uv, v = sampling_geometry(self.cfg)
        self.assertEqual(uv.shape, (8 * 62, 2))
        self.assertEqual(len(angles), 25)
        np.testing.assert_allclose(uv[0], [(2 + 0.5 - 155) * 2 / 310, (224 - 252) * 2 / 310])
        mask = acquisition_mask(np.rad2deg(angles), 0, 120)
        self.assertEqual(mask.sum(), 17)
        raw = np.ones((25, 8, 62)) * 3
        a, scale = normalize_acquired(raw, mask)
        raw[~mask] = 1e9
        b, changed = normalize_acquired(raw, mask)
        self.assertEqual(scale, changed)
        np.testing.assert_array_equal(a[mask], b[mask])

    def test_anisotropic_dictionary_consistency(self):
        cfg = dict(self.cfg, components=1, initialization_grid_xy=3)
        angles, uv, v = sampling_geometry(cfg)
        y = ray_basis(np.zeros((1, 3)), np.array([[0.16, 0.16, 0.12]]), angles, uv)[0].ravel() * 0.7
        reference, info = initialize_worm(y, angles, uv, v, cfg)
        np.testing.assert_allclose(reference[:3], 0, atol=1e-12)
        np.testing.assert_allclose(np.exp(reference[3:6]), [0.16, 0.16, 0.12], atol=1e-12)
        self.assertAlmostEqual(reference[-1], 0.7, places=10)
        self.assertLess(info["acquired_relative_residual"], 1e-10)


if __name__ == "__main__":
    unittest.main()
