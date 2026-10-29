"""Scientific isolation, mask identity, paired schedules and ranking controls."""

import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.worm_uncertainty.data.acquisition_diversity import (
    inventory,
    validate_inventory,
    mask_hash,
    load_role,
    schedule,
    preserve,
    score,
    risk_curve,
    disagreement,
)
from experiments.worm_uncertainty.data.completion_data import (
    channels,
    angular_extension,
    interpolate_missing,
)
from experiments.worm_uncertainty.core.completion_model import CompletionCNN
from experiments.worm_uncertainty.studies.acquisition_diversity import missing_loss


class DiversityTests(unittest.TestCase):
    def test_mask_identity_and_holdouts(self):
        inv = inventory()
        validate_inventory(inv)
        r = inv["B"][0]
        self.assertEqual(mask_hash(r["mask"]), r["hash"])
        inv["audit"].append(inv["B"][0])
        with self.assertRaises(ValueError):
            validate_inventory(inv)

    def test_reserved_rejected_before_io(self):
        d = {"roles": {"selection": ["secret.tiff"]}}
        m = {"files": [dict(name="secret.tiff", role="reserved")]}
        with patch("experiments.worm_uncertainty.data.acquisition_diversity.load_clean") as io:
            with self.assertRaises(ValueError):
                load_role("/absent", d, m, "selection")
            with self.assertRaises(ValueError):
                load_role("/absent", d, m, "reserved")
            io.assert_not_called()

    def test_balanced_paired_schedule(self):
        inv = inventory()
        a = schedule(768, 64, 20, 12, inv["A"])
        b = schedule(768, 64, 20, 12, inv["B"])
        c = schedule(768, 64, 20, 12, inv["C"])
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[2], b[2])
        np.testing.assert_array_equal(b[1], c[1])
        counts = np.array([inv["B"][i]["count"] for i in b[1].ravel()])
        self.assertEqual(len(set(np.unique(counts, return_counts=True)[1])), 1)
        np.testing.assert_array_equal(b[0], schedule(768, 64, 20, 12, inv["B"])[0])

    def test_hidden_targets_cannot_change_input(self):
        row = inventory()["B"][0]
        mask = np.array(row["mask"], bool)
        a = np.ones((25, 62))
        b = a.copy()
        b[~mask] = 1e9
        np.testing.assert_array_equal(
            channels(a, mask, row["start"], row["span"]),
            channels(b, mask, row["start"], row["span"]),
        )
        np.testing.assert_array_equal(interpolate_missing(a, mask), interpolate_missing(b, mask))
        np.testing.assert_array_equal(angular_extension(a)[:25], a[:, ::-1])

    def test_preserve_and_regenerate(self):
        rows = inventory()["A"][:1]
        mask = np.array(rows[0]["mask"], bool)
        observed = np.ones((1, 1, 8, 25, 62)) * 2
        target = np.ones((1, 8, 25, 62))
        pred = preserve(np.zeros_like(observed), observed, rows)
        np.testing.assert_array_equal(pred[0, 0, :, mask], observed[0, 0, :, mask])
        result = score(pred, target, rows, ["x"])
        self.assertEqual(result["groups"]["all"]["mean"], 1)
        self.assertEqual(result["groups"]["all"]["mse"], 1)

    def test_loss_normalizes_each_sample(self):
        pred = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]], requires_grad=True
        )
        mask = torch.tensor([[[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
        loss = missing_loss(pred, torch.zeros_like(pred), mask)
        self.assertAlmostEqual(float(loss), 2.5)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_batch_independence(self):
        torch.manual_seed(14)
        model = CompletionCNN().eval()
        x = torch.randn(3, 7, 25, 62)
        torch.testing.assert_close(model(x[:1]), model(x)[:1], atol=1e-6, rtol=1e-5)

    def test_ensemble_and_risk(self):
        error = np.arange(1.0, 101.0)
        self.assertLess(risk_curve(error, error)[0], risk_curve(error, -error)[0])
        row = inventory()["A"][:1]
        clean = np.ones((1, 8, 25, 62))
        pred = np.stack(
            [
                np.zeros((1, 1, 8, 25, 62)),
                np.ones((1, 1, 8, 25, 62)),
                np.ones((1, 1, 8, 25, 62)) * 2,
            ]
        )
        mean, std, _ = disagreement(pred, clean, row, ["x"])
        np.testing.assert_array_equal(mean, 1)
        np.testing.assert_array_equal(std, 1)


if __name__ == "__main__":
    unittest.main()
