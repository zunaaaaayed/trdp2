import unittest
import numpy as np
from phantom_variability import make_phantom, summarize


class PhantomTests(unittest.TestCase):
    def test_target_features_and_support(self):
        target, support, patch = make_phantom()
        self.assertEqual(target.shape, (64, 64))
        self.assertEqual(target[45, 29], 1.2)
        self.assertEqual(target[19, 39], 0.1)
        self.assertTrue((target[~support] == 0).all())
        self.assertGreater(np.std(target[patch]), 0)

    def test_bias_variance_separation(self):
        truth = np.zeros((2, 2))
        mask = np.ones((2, 2), dtype=bool)
        stack = np.stack([np.ones((2, 2)), np.ones((2, 2)) * 3])
        stats = summarize(stack, truth, mask)
        self.assertAlmostEqual(stats["rmse"], np.sqrt(5))
        self.assertAlmostEqual(stats["mean_bias_rmse"], 2)
        self.assertLess(stats["bias_variance_identity_error"], 1e-12)
        biased = summarize(np.ones((3, 2, 2)), truth, mask)
        self.assertEqual(biased["mean_noise_std"], 0)
        self.assertEqual(biased["fraction_bias_exceeding_two_noise_std"], 1)
