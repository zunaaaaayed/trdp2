import unittest
import json
from pathlib import Path
import numpy as np
import torch
from experiments.worm_uncertainty.core.completion_model import CompletionCNN, angular_connectivity
from experiments.worm_uncertainty.data.matched_completion import (
    read_protocol,
    grid,
    make_observation,
    make_channels,
    preserve_measured,
)
from experiments.worm_uncertainty.data.completion_data import acquisition_mask, interpolate_missing
from experiments.worm_uncertainty.studies.worm_shape_pilot import sampling_geometry
from experiments.worm_uncertainty.layout import resource_path


class MatchedTests(unittest.TestCase):
    def setUp(self):
        self.design, _ = read_protocol()

    def test_inherited_grid(self):
        cfg = json.loads(resource_path("worm_shape_protocol_v1.json").read_text())
        a, uv, v = sampling_geometry(cfg)
        degrees, u, w = grid(self.design)
        np.testing.assert_allclose(degrees, np.rad2deg(a), atol=1e-13)
        np.testing.assert_array_equal(u, uv)
        np.testing.assert_array_equal(w, v)
        np.testing.assert_allclose(u[:62, 0], -u[:62, 0][::-1], atol=1e-15)
        self.assertEqual(cfg["angle_indices"], list(range(0, 100, 4)))
        self.assertEqual(cfg["detector_indices"], list(range(2, 310, 5)))

    def test_shared_noise_and_withheld_isolation(self):
        d = self.design
        c = np.ones((25, 62))
        args = (d, "worm_153_gt.tiff", 256, 0, "validation")
        y, m = make_observation(c, *args, 0, 120)
        q, n = make_observation(c, *args, 36, 120)
        np.testing.assert_array_equal(y[m & n], q[m & n])
        changed = c.copy()
        changed[~m] = 1e9
        y2, m2 = make_observation(changed, *args, 0, 120)
        np.testing.assert_array_equal(y, y2)
        for arm in ["mask", "geometry"]:
            x = make_channels(y, m, 0, 120, arm)
            np.testing.assert_array_equal(x[0], y.astype(np.float32))
            np.testing.assert_array_equal(x[1, :, 0], m)
        pred = interpolate_missing(y, m)
        kept = preserve_measured(pred, y, m)
        np.testing.assert_array_equal(kept[m], y[m])
        reference = y[m].ravel().copy()
        for proposed in [pred, np.zeros_like(pred), np.ones_like(pred)]:
            np.testing.assert_array_equal(preserve_measured(proposed, y, m)[m].ravel(), reference)

    def test_artifact_mask_name_collision(self):
        from experiments.worm_uncertainty.data.matched_completion import evaluation_masks

        d = self.design
        arrays = {"identities": np.array([[0, 0, 0]]), "mask": np.ones((1, 25, 62))}
        recovered = evaluation_masks(arrays, d, d["familiar_masks"])
        self.assertEqual(recovered.dtype, bool)
        self.assertEqual(recovered.shape, (1, 25))
        self.assertEqual(int(recovered.sum()), 17)
        arrays["acquired_mask"] = ~recovered
        with self.assertRaises(ValueError):
            evaluation_masks(arrays, d, d["familiar_masks"])

    def test_context_and_capacity(self):
        torch.set_num_threads(1)
        torch.manual_seed(2026102803)
        model = CompletionCNN()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 5809)
        degrees, _, _ = grid(self.design)
        access = angular_connectivity()
        for start, span in self.design["familiar_masks"] + self.design["shift_masks"]:
            m = acquisition_mask(start, span, degrees)
            self.assertTrue(access[~m][:, m].any(axis=1).all())
            x = torch.ones(1, 7, 25, 62, requires_grad=True)
            with torch.no_grad():
                x[:, 0, ~m] = 0
                x[:, 1] = torch.tensor(m[:, None])
            out = model(x)
            for r in np.flatnonzero(~m):
                grad = torch.autograd.grad(out[0, r, 31], x, retain_graph=True)[0][0, 0]
                self.assertGreater(float(grad[m].abs().sum()), 0)
        # Dense 100-row design really fails at the centre of a 90-degree end gap.
        m = acquisition_mask(0, 90)
        self.assertFalse(angular_connectivity(100)[~m][:, m].any(axis=1).all())


if __name__ == "__main__":
    unittest.main()
