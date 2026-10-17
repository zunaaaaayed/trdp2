import unittest
import numpy as np
from compare_noise import masked_fbp
from check_fbp import reconstruct
from prepare import angle_mask


class MaskingTests(unittest.TestCase):
    def test_withheld_values_cannot_affect_reconstruction(self):
        angles, mask=angle_mask(0,90)
        a=np.random.default_rng(2).normal(size=(100,32))
        b=a.copy(); b[~mask]=1e8
        np.testing.assert_array_equal(masked_fbp(a,mask,angles),masked_fbp(b,mask,angles))

    def test_full_angle_and_zero_input(self):
        angles, mask=angle_mask(0,180)
        a=np.random.default_rng(3).normal(size=(100,32))
        np.testing.assert_array_equal(masked_fbp(a,mask,angles),reconstruct(a,angles))
        np.testing.assert_array_equal(masked_fbp(a,np.zeros(100,dtype=bool),angles),np.zeros((32,32)))
