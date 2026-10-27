"""Noise identity and equal-size angular windows for paired replication."""

import unittest
import numpy as np
from phantom_replication import acquisition_mask, observation


class ReplicationTests(unittest.TestCase):
    def test_equal_views_and_wrapping(self):
        degrees = np.arange(25) * 7.2
        for offset in [0, 28.8, 57.6, 172.8]:
            self.assertEqual(int(acquisition_mask(degrees, offset).sum()), 17)
            np.testing.assert_array_equal(
                acquisition_mask(degrees, offset), acquisition_mask(degrees, offset + 180)
            )

    def test_identical_noise_on_shared_rays(self):
        clean = np.arange(1, 101, dtype=float).reshape(25, 4)
        degrees = np.arange(25) * 7.2
        a, b = acquisition_mask(degrees, 0), acquisition_mask(degrees, 28.8)
        ya, sa = observation(clean, a, 7, 0.05)
        yb, sb = observation(clean, b, 7, 0.05)
        fulla = np.zeros_like(clean)
        fullb = np.zeros_like(clean)
        fulla[a] = (ya.reshape(-1, 4) - clean[a]) / sa
        fullb[b] = (yb.reshape(-1, 4) - clean[b]) / sb
        np.testing.assert_allclose(fulla[a & b], fullb[a & b], atol=1e-13)
        repeated, _ = observation(clean, a, 7, 0.05)
        np.testing.assert_array_equal(ya, repeated)
        exact, _ = observation(clean, a, None, 0.05)
        np.testing.assert_array_equal(exact, clean[a].ravel())


if __name__ == "__main__":
    unittest.main()
