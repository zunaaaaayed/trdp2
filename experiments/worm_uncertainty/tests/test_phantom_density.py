"""Known-truth controls for the phantom benchmark and acquired-only warm starts."""

import unittest
import json
from pathlib import Path
import numpy as np
import torch
from experiments.worm_uncertainty.core.voxel_parallel import ParallelProjector
from experiments.worm_uncertainty.core.completion_model import CompletionCNN
from experiments.worm_uncertainty.data.completion_data import channels, acquisition_mask
from experiments.worm_uncertainty.data.phantom_density import objects, density_metrics
from experiments.worm_uncertainty.studies.phantom_density import calibration
from experiments.worm_uncertainty.studies.acquisition_diversity import missing_loss


class PhantomTests(unittest.TestCase):
    def test_splits_and_exact_metrics(self):
        a = objects(dict(seed_base=810000, count=4))
        b = objects(dict(seed_base=820000, count=4))
        self.assertFalse(np.array_equal(a, b))
        np.testing.assert_array_equal(a, objects(dict(seed_base=810000, count=4)))
        for x in a:
            r = density_metrics(x, x)
            for key in [
                "density_l2",
                "density_rmse",
                "foreground_l2",
                "boundary_um",
                "interior_bias",
                "interior_mae",
            ]:
                self.assertEqual(r[key], 0)
            self.assertTrue(r["boundary_defined"])
        self.assertIsNone(density_metrics(np.zeros_like(a[0]), a[0])["boundary_um"])

    def test_warm_sart_uses_only_acquired(self):
        op = ParallelProjector(20, np.arange(25) * 7.2, spacing=5)
        truth = objects(dict(seed_base=810000, count=1))[:, :20, :20]
        y = op.forward(truth)
        mask = np.arange(25) < 13
        zero = op.sart(y, mask, initial=np.zeros_like(truth))[0]
        np.testing.assert_array_equal(zero, op.sart(y, mask)[0])
        initial = np.ones_like(truth) * 0.001
        damaged = y.copy()
        damaged[:, ~mask] = np.nan
        a = op.sart(y, mask, initial=initial)[0]
        b = op.sart(damaged, mask, initial=initial)[0]
        np.testing.assert_array_equal(a, b)
        self.assertGreaterEqual(a.min(), 0)
        np.testing.assert_array_equal(a[:, ~op.support], 0)

    def test_geometry_and_cpu_gradient_smoke(self):
        cfg = json.loads(
            Path("experiments/worm_uncertainty/protocols/phantom_density_v1.json").read_text()
        )
        op = ParallelProjector(62, np.arange(25) * 7.2, spacing=5)
        self.assertTrue(calibration(op, cfg)["passed"])
        torch.set_num_threads(1)
        torch.manual_seed(850001)
        truth = objects(dict(seed_base=810000, count=4))
        clean = op.forward(truth)
        mask = acquisition_mask(0, 90, np.arange(25) * 7.2)
        observed = np.where(mask[None, :, None], clean, 0)
        x = torch.tensor(np.stack([channels(v, mask, 0, 90) for v in observed]))
        model = CompletionCNN()
        loss = missing_loss(model(x), torch.tensor(clean, dtype=torch.float32), x[:, 1])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
