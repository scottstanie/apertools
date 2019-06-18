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
        self.im = np.array([[.1, 0.01, 2], [3, 4, 1 + 1j]])

    def test_clip(self):
        self.assertTrue(np.all(utils.clip(self.im) == np.array([[.1, 0.01, 1], [1, 1, 1]])))

    def test_log(self):
        out = np.array([[-20., -40, 6.020599], [9.542425, 12.041199, 3.010299]])
        self.assertTrue(np.allclose(out, utils.log(np.abs(self.im))))

    def test_looks(self):
        downsampled = utils.take_looks(self.im, 2, 1)
        assert_array_equal(downsampled, np.array([[1.55, 2.005, 1.5 + 0.5j]]))
        downsampled = utils.take_looks(self.im, 1, 2)
        assert_array_equal(downsampled, np.array([[0.055], [3.5]]))

    def test_mkdir_p(self):
        try:
            temp_dir = tempfile.mkdtemp()
            dir1 = os.path.join(temp_dir, 'd1')
            utils.mkdir_p(dir1)
            self.assertTrue(os.path.exists(dir1))
            utils.mkdir_p(dir1)
            self.assertTrue(os.path.exists(dir1))
        finally:
            shutil.rmtree(temp_dir)
            self.assertFalse(os.path.exists(dir1))

    def test_which(self):
        # Make sure to get the first os.environ["PATH"] has /bin first for test
        # Test direct path
        try:
            temp_dir = tempfile.mkdtemp()
            os.environ["PATH"] += os.pathsep + temp_dir
            test_exe = 'test_exe'
            test_exe_full = os.path.join(temp_dir, test_exe)
            # Create temp file,
            open(test_exe_full, 'a').close()
            # Make it executable
            st = os.stat(test_exe_full)
            os.chmod(test_exe_full, st.st_mode | stat.S_IEXEC)

            self.assertEqual(utils.which(test_exe), test_exe_full)
            self.assertEqual(utils.which(test_exe_full), test_exe_full)

            self.assertIsNone(utils.which('non_existent_program'))
        finally:
            shutil.rmtree(temp_dir)
