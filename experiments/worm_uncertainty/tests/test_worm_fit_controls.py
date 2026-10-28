"""Checks for dictionary consistency, label matching and failed-fit handling."""

import unittest
import numpy as np
from experiments.worm_uncertainty.core.cholesky_gaussian import projection
from experiments.worm_uncertainty.studies.worm_fit_controls import (
    align_to_reference,
    multiscale_initialize,
    stability,
)


class WormFitControlTests(unittest.TestCase):
    def test_multiscale_density_projection_matches_reported_residual(self):
        angles = np.array([0.0, 0.5, 1.0])
        uv = np.array([[u, z] for z in [-0.1, 0.0, 0.1] for u in np.linspace(-0.8, 0.8, 15)])
        truth = np.r_[[0.0, 0.0, 0.0], np.log([0.08, 0.08, 0.12]), np.zeros(3), 3.0]
        y = projection(truth, angles, uv)[0]
        cfg = dict(
            components=1,
            initialization_extent=0.8,
            initialization_grid_z=3,
            initialization_width=0.16,
            initialization_width_z=0.12,
            density_bounds=[0.001, 100.0],
        )
        design = dict(
            grid_xy=5, widths_xy=[0.08, 0.16], minimum_centre_distance=0.04, dictionary_chunk=7
        )
        p, info = multiscale_initialize(y, angles, uv, np.array([-0.1, 0.1]), cfg, design)
        np.testing.assert_allclose(projection(p, angles, uv)[0], y, atol=1e-12)
        self.assertLess(info["realized_relative_residual"], 1e-12)
        self.assertEqual(info["active_densities"], 1)

    def test_label_matching_preserves_projection_and_restores_reference_order(self):
        p = np.r_[
            [-0.2, 0, 0, 0.2, 0.1, 0],
            np.log([0.1, 0.2, 0.12, 0.15, 0.08, 0.11]),
            np.zeros(6),
            [2.0, 3.0],
        ]
        swapped = np.concatenate(
            [p[i * 2 : (i + 3) * 2].reshape(2, 3)[::-1].ravel() for i in [0, 3, 6]] + [p[18:][::-1]]
        )
        matched = align_to_reference(swapped, p)
        np.testing.assert_array_equal(matched, p)
        args = (np.array([0.1, 0.5]), np.array([[0.0, 0.0], [0.1, 0.1]]))
        np.testing.assert_allclose(projection(swapped, *args)[0], projection(matched, *args)[0])

    def test_unfinished_fit_is_provisional_only(self):
        rows = [dict(seed=0, converged=False, objective=1.0, active_indices=[], parameters=[])]
        chosen, gate = stability(rows, np.array([]), np.empty((0, 2)), dict(minimum_converged=2))
        self.assertFalse(gate["passed"])
        self.assertEqual(chosen["seed"], 0)

    def test_convergence_alone_does_not_pass_stability(self):
        p = np.r_[np.zeros(3), np.log([0.1, 0.1, 0.12]), np.zeros(3), 2.0]
        first = dict(seed=0, converged=True, objective=1.0, active_indices=[], parameters=p)
        second = dict(first, seed=81)
        args = (
            np.array([0.0]),
            np.array([[0.0, 0.0]]),
            dict(
                minimum_converged=2,
                relative_objective_gap=1e-5,
                relative_prediction_disagreement=1e-4,
            ),
        )
        self.assertTrue(stability([first, second], *args)[1]["passed"])
        self.assertFalse(stability([first, dict(second, active_indices=[9])], *args)[1]["passed"])
        self.assertFalse(stability([first, dict(second, objective=1.1)], *args)[1]["passed"])
        changed = p.copy()
        changed[-1] *= 1.1
        self.assertFalse(stability([first, dict(second, parameters=changed)], *args)[1]["passed"])


if __name__ == "__main__":
    unittest.main()
