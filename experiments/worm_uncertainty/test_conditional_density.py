import unittest
import numpy as np
import torch
from conditional_density import (
    covariance_from_parameters,
    projection_basis,
    voxel_basis,
    solve_posterior,
    coverage_summary,
)
from native_parallel import analytic_projection


class DensityPosteriorTests(unittest.TestCase):
    def test_axis_aligned_analytic_integrals(self):
        xyz = np.array([[0.1, -0.2, 0.15]])
        scales = np.array([[0.2, 0.3, 0.1]])
        cov = covariance_from_parameters(scales, np.array([[1.0, 0, 0, 0]]))
        coord = (np.arange(16) + 0.5) * 2 / 16 - 1
        u, v = np.meshgrid(coord, coord)
        result = projection_basis(
            torch.tensor(xyz),
            torch.tensor(cov),
            0.7,
            torch.tensor(np.stack([u.ravel(), v.ravel()], 1)),
        )
        np.testing.assert_allclose(
            result.numpy().reshape(16, 16),
            analytic_projection(xyz, scales, [1], 0.7, 16),
            atol=1e-12,
        )

    def test_rotated_covariance_and_ray_quadrature(self):
        xyz = np.array([[0.1, -0.05, 0.02]])
        scales = np.array([[0.2, 0.3, 0.1]])
        cov = covariance_from_parameters(scales, np.array([[0.8, 0.1, 0.3, 0.2]]))
        angle = 0.6
        u, v = 0.15, -0.1
        t = np.linspace(-3, 3, 10001)
        direction = np.array([-np.sin(angle), -np.cos(angle), 0])
        origin = np.array([u * np.cos(angle), -u * np.sin(angle), v])
        points = origin[None, :] + t[:, None] * direction
        values = voxel_basis(torch.tensor(xyz), torch.tensor(cov), torch.tensor(points)).numpy()[
            :, 0
        ]
        exact = projection_basis(
            torch.tensor(xyz), torch.tensor(cov), angle, torch.tensor([[u, v]], dtype=torch.float64)
        ).item()
        self.assertAlmostEqual(np.trapz(values, t), exact, places=10)

    def test_posterior_matches_direct_and_null_direction(self):
        a = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
        y = torch.tensor([1.0, 2.0], dtype=torch.float64)
        m, c = solve_posterior(a.T @ a, a.T @ y, 0.1, 2.0)
        self.assertAlmostEqual(c[1, 1].item(), 4.0)
        self.assertAlmostEqual(m[1].item(), 0.0)
        expected = torch.linalg.inv(a.T @ a / 0.01 + torch.eye(2, dtype=torch.float64) / 4)
        np.testing.assert_allclose(c.numpy(), expected.numpy(), atol=1e-12)

    def test_latent_vs_predictive_width(self):
        truth = np.zeros(4)
        pred = np.ones(4)
        selected = np.ones(4, dtype=bool)
        self.assertEqual(coverage_summary(pred, np.ones(4) * 0.1, truth, selected)["coverage95"], 0)
        self.assertEqual(coverage_summary(pred, np.ones(4), truth, selected)["coverage95"], 1)
