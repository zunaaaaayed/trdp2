"""Scientific controls for the dense-grid self-supervised reconstruction adapter."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.worm_uncertainty.core.voxel_parallel import ParallelProjector, relative_l2
from experiments.worm_uncertainty.data.alternative_reconstruction import (
    allowed_record,
    noisy_pair,
    MANIFEST,
)
from experiments.worm_uncertainty.core.icecream_adapter import (
    canonical_axes,
    native_axes,
    wedge,
    rotation_set,
    rotate,
    infer_pair,
    make_trainer,
    load_official,
)


class OperatorTests(unittest.TestCase):
    def test_exact_transpose_and_reversal(self):
        rng = np.random.default_rng(1)
        op = ParallelProjector(24, [0, 17, 90, 133])
        x, y = rng.normal(size=(3, 24, 24)), rng.normal(size=(3, 4, 24))
        np.testing.assert_allclose(np.vdot(op.forward(x), y), np.vdot(x, op.adjoint(y)), rtol=1e-13)
        shifted = ParallelProjector(24, [180, 197, 270, 313])
        np.testing.assert_allclose(shifted.forward(x), op.forward(x)[..., ::-1], atol=1e-12)

    def test_sart_uses_acquired_only_and_reduces_residual(self):
        op = ParallelProjector(20, np.arange(10) * 18.0)
        x = np.zeros((1, 20, 20))
        x[:, 7:12, 8:14] = 1
        y = op.forward(x)
        mask = np.arange(10) < 6
        a, history = op.sart(y, mask, sweeps=3)
        bad = y.copy()
        bad[:, ~mask] = np.nan
        b, _ = op.sart(bad, mask, sweeps=3)
        np.testing.assert_array_equal(a, b)
        self.assertLess(history[-1], history[0])
        self.assertGreaterEqual(a.min(), 0)
        np.testing.assert_array_equal(op.fbp(y, mask, 18), op.fbp(bad, mask, 18))

    def test_fixed_spacing_linear_scale_and_mass(self):
        op = ParallelProjector(32, [0, 45, 90], spacing=2)
        x = np.zeros((1, 32, 32))
        x[:, 12:18, 13:19] = 1
        y = op.forward(x)
        np.testing.assert_allclose(y.sum(-1) * op.du, x.sum() * op.spacing**2)
        np.testing.assert_allclose(
            op.fbp(y * 3, np.ones(3, bool), 60), 3 * op.fbp(y, np.ones(3, bool), 60), atol=1e-14
        )


class InformationTests(unittest.TestCase):
    def test_reserved_refused_before_io(self):
        manifest = json.loads(MANIFEST.read_text())
        for row in manifest["files"]:
            if row["role"] == "reserved":
                with self.assertRaises(ValueError):
                    allowed_record(row["name"])

    def test_independent_noise_and_average_variance(self):
        x = np.zeros((50, 100, 100))
        mask = np.arange(100) < 60
        pair = noisy_pair(x, mask, 0.05, [1, 2])
        noise = pair[:, :, mask, :].reshape(2, -1)
        self.assertLess(abs(np.corrcoef(noise)[0, 1]), 0.01)
        self.assertLess(abs(np.var(noise.mean(0)) / (0.05**2 / 2) - 1), 0.015)
        np.testing.assert_array_equal(pair, noisy_pair(x, mask, 0.05, [1, 2]))
        x[:, ~mask] = np.nan
        np.testing.assert_array_equal(pair, noisy_pair(x, mask, 0.05, [1, 2]))
        with self.assertRaises(ValueError):
            noisy_pair(x, mask, 0.05, [1, 1])

    def test_axis_roundtrip(self):
        v = np.arange(4 * 5 * 6).reshape(4, 5, 6)
        np.testing.assert_array_equal(v, canonical_axes(native_axes(v)))


class IcecreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        try:
            load_official()
        except ImportError as exc:
            raise unittest.SkipTest("isolated ICECREAM environment needed: " + str(exc))

    def test_wedge_orientation_and_reflections(self):
        w = wedge(32, 0, 90, spherical=False)
        # (kx positive, ky negative) is observed; opposite diagonal is missing.
        self.assertEqual(float(w[20, 12, 16]), 1)
        self.assertEqual(float(w[20, 20, 16]), 0)
        np.testing.assert_array_equal(w.numpy(), np.flip(w.numpy(), (0, 1, 2)))
        # Physical z has no wedge angular effect (without spherical support).
        np.testing.assert_array_equal(w[:, :, 5], w[:, :, 25])
        np.testing.assert_array_equal(w, wedge(32, 180, 90, spherical=False))
        for k in rotation_set(w):
            self.assertGreater(
                float(torch.linalg.vector_norm(rotate(w, k) - w) / torch.linalg.vector_norm(w)), 0.5
            )
            label = torch.arange(27).reshape(3, 3, 3)
            np.testing.assert_array_equal(
                torch.sort(rotate(label, k).ravel()).values, torch.arange(27)
            )

    def test_adapted_wedge_matches_upstream_in_supported_orientation(self):
        from icecream.utils.utils import get_wedge_3d_new

        # Symmetric +/-60 acquired interval; avoid angular boundary samples.
        ours = wedge(48, -60, 120).numpy()
        original, _ = get_wedge_3d_new(48, 60, -60, use_spherical_support=True)
        # Upstream broadcasts its 2D mask along axis 0; our CT rotation axis is 2.
        original = original.transpose(1, 2, 0)
        # Upstream excludes the central Fourier line; our explicit DC inclusion is needed.
        original[24, 24, :] = ours[24, 24, :]
        q = np.linspace(-1, 1, 49)
        xyz = np.meshgrid(q, q, q, indexing="ij")
        away_from_sphere_boundary = np.abs(sum(a**2 for a in xyz) - 1) > 1e-12
        np.testing.assert_array_equal(
            ours[away_from_sphere_boundary], original[away_from_sphere_boundary]
        )

    def test_crop_merger_identity_and_scale(self):
        pair = [torch.randn(19, 23, 21) for _ in range(2)]
        trainer = SimpleNamespace(
            model=torch.nn.Identity(),
            device=torch.device("cpu"),
            window=torch.zeros(8, 8, 8),
            wedge_input_set=[torch.ones(16, 16, 16)],
        )
        trainer.window[2:6, 2:6, 2:6] = 1
        out = infer_pair(trainer, pair, 8)
        np.testing.assert_allclose(out, ((pair[0] + pair[1]) / 2).numpy(), atol=1e-6)

    def test_official_loss_finite_gradients_and_no_targets(self):
        trainer, _ = make_trainer(32, 0, 120)
        a = torch.randn(1, 32, 32, 32) * 0.1
        b = torch.randn_like(a) * 0.1
        loss = trainer.compute_loss(a, b, 0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
        self.assertIn("icecream.trainer.eq_trainer", trainer.compute_loss.__module__)


if __name__ == "__main__":
    unittest.main()
