"""Conditional optimum, active constraints and shared-objective regression checks."""

import unittest
import numpy as np
from experiments.worm_uncertainty.core.staged_density import solve_density, fit_stages
from experiments.worm_uncertainty.core.cholesky_gaussian import fit_control, projection


class StagedDensityTests(unittest.TestCase):
    def test_interior_matches_closed_form(self):
        b = np.array([[1.0, 0.2], [0.3, 1.0], [0.7, 0.4]])
        y = b @ np.array([2.0, 3.0])
        a, info = solve_density(b, y, 0.05, 10.0, [0.001, 100.0])
        expected = np.linalg.solve(b.T @ b / 0.05**2 + np.eye(2) / 100, b.T @ y / 0.05**2)
        np.testing.assert_allclose(a, expected, atol=1e-11)
        self.assertLess(info["kkt_relative"], 1e-10)

    def test_both_bounds_and_rank_deficiency(self):
        a, info = solve_density(np.eye(2), np.array([-2.0, 200.0]), 1.0, 10.0, [0.001, 100.0])
        np.testing.assert_allclose(a, [0.001, 100.0])
        self.assertEqual(info["active_indices"], [0, 1])
        b = np.ones((3, 2))
        a, _ = solve_density(b, np.ones(3), 0.1, 10.0, [0.001, 100.0])
        self.assertAlmostEqual(a[0], a[1], places=10)

    def test_joint_matches_existing_solver_and_stages_decrease(self):
        p = np.r_[[0.1, -0.1, 0.02], np.log([0.15, 0.1, 0.12]), np.zeros(3), 2.0]
        cfg = dict(
            position_prior_sd=0.3,
            cholesky_log_diagonal_prior_sd=0.5,
            cholesky_off_diagonal_prior_sd=0.1,
            density_prior_sd=10.0,
            position_bounds=[-0.95, 0.95],
            density_bounds=[0.001, 100.0],
            noise_sigma=0.05,
            budget_nfev=30,
        )
        angles = np.array([0.0, 0.5, 1.0])
        uv = np.array([[u, z] for u in np.linspace(-0.5, 0.5, 10) for z in [-0.1, 0, 0.1]])
        y = projection(p, angles, uv)[0]
        initial = p.copy()
        initial[-1] = 1.0
        old, _ = fit_control(y, angles, uv, initial, p, 0.05, cfg, False)
        new = fit_stages(y, angles, uv, initial, p, cfg, [dict(kind="joint", budget=30)])
        np.testing.assert_allclose(new["parameters"], old.x, atol=1e-10)
        self.assertAlmostEqual(new["objective"], 2 * old.cost, places=10)
        stages = [
            dict(kind="density"),
            dict(kind="centre", budget=5),
            dict(kind="density"),
            dict(kind="geometry", budget=5),
            dict(kind="density"),
            dict(kind="joint", budget=20),
        ]
        fit = fit_stages(y, angles, uv, initial, p, cfg, stages)
        for t in fit["trace"]:
            self.assertLessEqual(t["after"]["objective"], t["before"]["objective"] + 1e-8)


if __name__ == "__main__":
    unittest.main()
