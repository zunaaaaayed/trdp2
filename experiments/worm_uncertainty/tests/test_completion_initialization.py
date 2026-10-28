import unittest
import numpy as np
from experiments.worm_uncertainty.core.completion_initialization import matched_contracts
from experiments.worm_uncertainty.studies.worm_fit_controls import align_to_reference
from experiments.worm_uncertainty.core.cholesky_gaussian import projection


class InitializationTests(unittest.TestCase):
    def test_shared_contract_copies(self):
        y = np.arange(24.0).reshape(4, 2, 3)
        m = np.array([1, 0, 1, 0], bool)
        a = np.arange(4.0)
        ref = np.arange(20.0)
        uv = np.zeros((6, 2))
        cfg = {"budget_nfev": 450, "noise_sigma": 0.05, "bounds": [-1, 1]}
        c = matched_contracts(y, m, a, uv, ref, cfg)
        for v in c.values():
            np.testing.assert_array_equal(v["y"], y[m].ravel())
            np.testing.assert_array_equal(v["reference"], ref)
            self.assertEqual(v["cfg"], cfg)
        c["learned"]["cfg"]["bounds"][0] = -99
        c["learned"]["y"][0] = 999
        self.assertEqual(c["acquired"]["cfg"]["bounds"][0], -1)
        self.assertEqual(c["acquired"]["y"][0], 0)

    def test_component_alignment_preserves_forward_model(self):
        p = np.r_[
            [0.2, 0.1, 0, -0.2, -0.1, 0],
            np.log([0.1, 0.2, 0.3, 0.15, 0.25, 0.35]),
            np.zeros(6),
            [1.0, 2.0],
        ]
        q = np.r_[
            p[:6].reshape(2, 3)[::-1].ravel(),
            p[6:12].reshape(2, 3)[::-1].ravel(),
            p[12:18].reshape(2, 3)[::-1].ravel(),
            p[18:][::-1],
        ]
        aligned = align_to_reference(q, p)
        np.testing.assert_array_equal(aligned, p)
        a = np.array([0.0, 0.8])
        uv = np.array([[0.0, 0.0], [0.1, 0.2]])
        np.testing.assert_allclose(projection(q, a, uv)[0], projection(aligned, a, uv)[0])


if __name__ == "__main__":
    unittest.main()
