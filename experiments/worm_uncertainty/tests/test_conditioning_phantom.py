"""Check controlled shape construction and the oracle prior caveat."""

import unittest
import numpy as np
from experiments.worm_uncertainty.studies.conditioning_phantom import phantom
from experiments.worm_uncertainty.core.cholesky_gaussian import geometry, projection


class ConditioningPhantomTests(unittest.TestCase):
    def test_matched_volume_and_shape_ratios(self):
        m, e = phantom(False), phantom(True)
        cm, sm, _, dm = geometry(m)
        ce, se, _, de = geometry(e)
        np.testing.assert_allclose(cm, ce)
        np.testing.assert_allclose(dm, de)
        np.testing.assert_allclose(np.linalg.det(sm), np.linalg.det(se), rtol=1e-10)
        self.assertAlmostEqual(
            float(np.sqrt(np.linalg.eigvalsh(se)[0, -1] / np.linalg.eigvalsh(se)[0, 0])),
            100.0,
            places=8,
        )
        self.assertTrue(
            np.isfinite(
                projection(e, np.array([0.0, 0.5]), np.array([[0.0, 0.0], [0.1, 0.1]]))[0]
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
