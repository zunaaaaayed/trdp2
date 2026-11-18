"""Synthetic-only controls; these do not score or load worm outcomes."""

import unittest
import numpy as np
import torch
from experiments.worm_uncertainty.core.voxel_parallel import ParallelProjector
from experiments.worm_uncertainty.core.differentiable_fbp import DifferentiableFBP, completion_loss


class FBPTests(unittest.TestCase):
    def test_reference_and_gradient_isolation(self):
        torch.manual_seed(71)
        op = ParallelProjector(24, np.arange(25) * 7.2)
        layer = DifferentiableFBP(op, 7.2)
        prediction = torch.randn(2, 25, 24, dtype=torch.float64, requires_grad=True)
        measured = torch.randn_like(prediction)
        mask = torch.arange(25) < 13
        completed = torch.where(mask[:, None], measured, prediction)
        np.testing.assert_allclose(
            layer(completed, mask).detach().numpy(),
            op.fbp(completed.detach().numpy(), mask.numpy(), 7.2),
            rtol=1e-11,
            atol=1e-12,
        )
        target = torch.randn_like(prediction)
        base = completion_loss(prediction, measured, mask, target, layer, 0)
        expected = (prediction[:, ~mask] - target[:, ~mask]).square().mean()
        torch.testing.assert_close(base, expected)
        loss = completion_loss(prediction, measured, mask, target, layer, 1)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(float(prediction.grad[:, mask].abs().max()), 0)
        self.assertGreater(float(prediction.grad[:, ~mask].abs().sum()), 0)
        np.testing.assert_array_equal(
            completed.detach().numpy()[:, mask], measured.numpy()[:, mask]
        )


if __name__ == "__main__":
    unittest.main()
