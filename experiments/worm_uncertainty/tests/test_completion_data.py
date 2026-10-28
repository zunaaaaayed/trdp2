from experiments.worm_uncertainty.layout import resource_path
import unittest
import json
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.data.completion_data import (
    acquisition_mask,
    angular_extension,
    interpolate_missing,
    observation,
    validate_manifest,
)
from experiments.worm_uncertainty.core.posterior_local import projection_mean


class TestCompletion(unittest.TestCase):
    def test_mask_noise_reproducible(self):
        clean = np.ones((100, 310))
        x, m = observation(clean, 2, 0.05, 7, 0, 120)
        y, n = observation(clean, 2, 0.05, 7, 36, 120)
        np.testing.assert_array_equal(x[0, m & n], y[0, m & n])
        self.assertTrue((x[0, ~m] == 0).all())
        self.assertTrue((x[1, ~m] == 0).all())
        self.assertEqual(acquisition_mask(162, 36).sum(), 20)
        self.assertEqual(acquisition_mask(0, 180).sum(), 100)

    def test_withheld_isolation(self):
        c = np.ones((100, 310))
        x, m = observation(c, 2, 0.05, 7, 0, 120)
        c[~m] = 999999
        y, _ = observation(c, 2, 0.05, 7, 0, 120)
        np.testing.assert_array_equal(x, y)
        a = interpolate_missing(x[0], m)
        changed = x[0].copy()
        changed[~m] = np.nan
        np.testing.assert_array_equal(a, interpolate_missing(changed, m))
        np.testing.assert_array_equal(a[m], x[0, m])

    def test_parallel_wrap_against_analytic_projector(self):
        p = np.r_[0.2, 0.1, 0, np.log([0.1, 0.2, 0.3]), 0.0, 0.0, 0.0, 1.0]
        u = np.linspace(-1, 1, 31)
        uv = np.column_stack([u, np.zeros(len(u))])
        a = np.arange(10) * np.pi / 10
        y = projection_mean(p, a, uv).reshape(10, 31)
        full = projection_mean(p, np.r_[a - np.pi, a, a + np.pi], uv).reshape(30, 31)
        np.testing.assert_allclose(angular_extension(y), full, atol=1e-14)

    def test_manifest(self):
        m = json.loads(resource_path("completion_manifest_v1.json").read_text())
        validate_manifest(m)
        m["files"][1]["role"] = "reserved"
        with self.assertRaises(ValueError):
            validate_manifest(m)


if __name__ == "__main__":
    unittest.main()
