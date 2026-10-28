import unittest
import numpy as np
from experiments.worm_uncertainty.studies.limited_gaussian import (
    initialize_measured,
    projection_errors,
)


class LimitedGaussianTests(unittest.TestCase):
    def test_initializer_ignores_withheld(self):
        angles = np.arange(10) * 18.0
        mask = angles < 90
        data = np.ones((10, 2, 16))
        altered = data.copy()
        altered[~mask] = 1e8
        first = initialize_measured(data, angles, mask, 2 / 16, 12, 0)
        second = initialize_measured(altered, angles, mask, 2 / 16, 12, 0)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_error_partition(self):
        clean = np.ones((4, 2, 3))
        pred = clean.copy()
        pred[2:] *= 2
        stats = projection_errors(pred, clean, np.array([True, True, False, False]))
        self.assertEqual(stats["measured_relative_l2"], 0)
        self.assertEqual(stats["withheld_relative_l2"], 1)
