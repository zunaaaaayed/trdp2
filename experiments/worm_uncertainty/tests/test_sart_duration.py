"""SART continuation must preserve both schedule and measured-only information."""

import unittest
import numpy as np
from experiments.worm_uncertainty.core.voxel_parallel import ParallelProjector


class SartDurationTests(unittest.TestCase):
    def test_chunking_matches_total_budget_and_withheld_isolation(self):
        rng = np.random.default_rng(43)
        op = ParallelProjector(20, np.arange(12) * 15, spacing=5)
        truth = np.zeros((2, 20, 20))
        truth[:, 8:13, 7:11] = 0.01
        clean = op.forward(truth)
        mask = np.arange(12) < 7
        observed = np.where(mask[None, :, None], clean + rng.normal(0, 0.01, clean.shape), 0)
        start = np.maximum(rng.normal(0.001, 0.0002, truth.shape), 0) * op.support
        direct, _ = op.sart(observed, mask, sweeps=20, initial=start)
        chunked = start
        for sweeps in [1, 4, 5, 10]:
            chunked, _ = op.sart(observed, mask, sweeps=sweeps, initial=chunked)
        np.testing.assert_array_equal(chunked, direct)
        poisoned = observed.copy()
        poisoned[:, ~mask] = np.nan
        hidden, _ = op.sart(poisoned, mask, sweeps=20, initial=start)
        np.testing.assert_array_equal(hidden, direct)
        self.assertTrue(np.isfinite(direct).all())
        self.assertGreaterEqual(float(direct.min()), 0)
        np.testing.assert_array_equal(direct[:, ~op.support], 0)


if __name__ == "__main__":
    unittest.main()
