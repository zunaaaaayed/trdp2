"""Acquired-only selection, equal-start accounting and inverse-view orientation."""

import json
import unittest
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.core.completion_evaluation import (
    common_starts,
    select_fit,
    filtered_backprojection,
    polar_spectrum,
)


class EvaluationTests(unittest.TestCase):
    def test_selection_ignores_withheld_scores(self):
        cfg = json.loads(
            (Path(__file__).parents[1] / "protocols/worm_shape_protocol_v1.json").read_text()
        )
        p = np.zeros(10)
        p[-1] = 1
        rows = [
            dict(
                arm="a",
                seed=i,
                parameters=p.tolist(),
                objective=obj,
                converged=True,
                numerical=dict(accepted=accepted),
                active_indices=[],
                withheld_error=withheld,
            )
            for i, obj, accepted, withheld in [
                (0, 1, False, 0),
                (1, 2, True, 100),
                (2, 3, True, 0.1),
            ]
        ]
        chosen, _ = select_fit(rows, p, cfg, np.array([0]), np.zeros((1, 2)))
        self.assertEqual(chosen["seed"], 1)
        rows[1]["withheld_error"] = 1e8
        self.assertEqual(select_fit(rows, p, cfg, np.array([0]), np.zeros((1, 2)))[0]["seed"], 1)
        self.assertEqual(
            common_starts(rows + [dict(arm="b", seed=2, parameters=p)], ["a", "b"]), [2]
        )

    def test_fbp_known_gaussian_orientation_and_scale(self):
        u = np.linspace(-1, 1, 257)
        angles = np.arange(180) * np.pi / 180
        xy = np.array([0.22, -0.14])
        cov = np.array([[0.012, 0.003], [0.003, 0.006]])
        e = np.column_stack([np.cos(angles), -np.sin(angles)])
        variance = np.einsum("ai,ij,aj->a", e, cov, e)
        centre = e @ xy
        sino = (
            1.5
            * np.sqrt(2 * np.pi * np.linalg.det(cov) / variance)[:, None]
            * np.exp(-0.5 * (u[None, :] - centre[:, None]) ** 2 / variance[:, None])
        )
        image = filtered_backprojection(sino, u, angles, 101)
        peak = np.array(np.unravel_index(image.argmax(), image.shape))
        recovered = np.array([np.linspace(-1, 1, 101)[peak[1]], np.linspace(-1, 1, 101)[peak[0]]])
        np.testing.assert_allclose(recovered, xy, atol=0.021)
        self.assertAlmostEqual(image.max(), 1.5, delta=0.06)
        fx, fy, s = polar_spectrum(sino, u, angles)
        self.assertEqual(fx.shape, sino.shape)
        self.assertTrue(np.isfinite(s).all())
        np.testing.assert_allclose(fy[0], 0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
