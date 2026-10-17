import unittest
import numpy as np
from prepare import angle_mask, noisy_copy


class PreparationTests(unittest.TestCase):
    def test_angles(self):
        angles, mask = angle_mask(0, 180)
        self.assertEqual(mask.sum(), 100)
        self.assertAlmostEqual(angles[-1], 178.2)
        _, mask = angle_mask(0, 90)
        self.assertEqual(mask.sum(), 50)
        _, wrapped = angle_mask(90, 90)
        np.testing.assert_array_equal(wrapped, ~mask)
        _, wrapped = angle_mask(178.2, 3.6)
        self.assertTrue(wrapped[99] and wrapped[0])
        self.assertEqual(wrapped.sum(), 2)

    def test_noise_pairing_and_source_preservation(self):
        clean = np.arange(1000, dtype=float)
        saved = clean.copy()
        a = noisy_copy(clean, 100, 0.01, 7)
        np.testing.assert_array_equal(a, noisy_copy(clean, 100, 0.01, 7))
        self.assertFalse(np.array_equal(a, noisy_copy(clean, 100, 0.01, 8)))
        np.testing.assert_array_equal(clean, saved)
        np.testing.assert_array_equal(clean, noisy_copy(clean, 100, 0, 7))

    def test_noise_statistics_and_no_clipping(self):
        values = noisy_copy(np.zeros(200000), 10, 0.2, 3)
        self.assertLess(abs(values.mean()), 0.02)
        self.assertAlmostEqual(values.std(), 2, delta=0.02)
        self.assertTrue((values < 0).any())

    def test_invalid_inputs(self):
        for start, span in [(180, 90), (0, 0), (0, 181), (float("nan"), 90)]:
            with self.assertRaises(ValueError):
                angle_mask(start, span)
        for scale, fraction, seed in [(0, 0.1, 0), (1, -0.1, 0), (1, 0.1, -1)]:
            with self.assertRaises(ValueError):
                noisy_copy(np.zeros(2), scale, fraction, seed)
        with self.assertRaises(ValueError):
            noisy_copy(np.array([np.nan]), 1, 0.1, 0)
