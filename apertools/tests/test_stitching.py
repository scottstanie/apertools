import unittest
# import os
# import stat
# import tempfile
# import shutil
import numpy as np
from numpy.testing import assert_array_equal
from apertools import stitching


class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.aa = np.ones((10, 1)) @ np.arange(10).reshape((1, 10))
        self.aa = self.aa.astype(np.complex64)

    def test_stitch(self):
        a1 = self.aa.copy()
        a2 = self.aa.copy()

        # Make two with two-row overlap
        a1[:4] = 0
        a2[6:] = 0
        out = stitching.combine_complex([a1, a2])
        assert_array_equal(out, self.aa)
