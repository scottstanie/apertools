import unittest
import os
import stat
import tempfile
import shutil
import numpy as np
from numpy.testing import assert_array_equal
from apertools import utils


class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.im = np.array([[0.1, 0.01, 2], [3, 4, 1 + 1j]])

    def test_db(self):
        out = np.array([[-20.0, -40, 6.020599], [9.542425, 12.041199, 3.010299]])
        self.assertTrue(np.allclose(out, utils.db(np.abs(self.im))))

    def test_looks(self):
        downsampled = utils.take_looks(self.im, 2, 1)
        assert_array_equal(downsampled, np.array([[1.55, 2.005, 1.5 + 0.5j]]))
        downsampled = utils.take_looks(self.im, 1, 2)
        assert_array_equal(downsampled, np.array([[0.055], [3.5]]))
