import unittest
# import os
# from os.path import join, dirname, exists
# import tempfile
# import shutil
import numpy as np
from numpy.testing import assert_array_almost_equal

from apertools import latlon


class TestLatlonConversion(unittest.TestCase):
    def setUp(self):
        # self.datapath = join(dirname(__file__), 'data')
        # self.rsc_path = join(self.datapath, 'elevation.dem.rsc')
        # self.dem_path = join(self.datapath, 'elevation.dem')

        self.im_test = np.arange(20).reshape((5, 4))
        self.rsc_info1 = {
            'x_first': -5.0,
            'y_first': 4.0,
            'x_step': 0.5,
            'y_step': -0.5,
            'file_length': 5,
            'width': 4
        }
        self.im1 = latlon.LatlonImage(data=self.im_test, rsc_data=self.rsc_info1)

        self.im_test2 = np.arange(30).reshape((6, 5))
        self.rsc_info2 = {
            'x_first': -4.0,
            'y_first': 2.5,
            'x_step': 0.5,
            'y_step': -0.5,
            'file_length': 6,
            'width': 5
        }
        self.im2 = latlon.LatlonImage(data=self.im_test2, rsc_data=self.rsc_info2)
        self.image_list = [self.im1, self.im2]

    def test_find_total_pixels(self):
        self.assertEqual((9, 7), latlon.find_total_pixels(self.image_list))

    def test_find_img_intersection(self):
        expected = (3, 2)
        self.assertEqual(expected, latlon.find_img_intersections(*self.image_list))

    def test_stitch(self):
        out = np.zeros(latlon.find_total_pixels(self.image_list))
        start_row, start_col = latlon.find_img_intersections(*self.image_list)
        out[start_row:, start_col:] = self.im2


class TestLatlonImage(unittest.TestCase):
    def setUp(self):
        self.im_data = np.arange(20).reshape((5, 4))
        self.stack_data = np.arange(60).reshape((3, 5, 4))
        self.rsc_info = {
            'x_first': -5.0,
            'y_first': 4.0,
            'x_step': 0.5,
            'y_step': -0.5,
            'file_length': 5,
            'width': 4
        }
        self.im_ll = latlon.LatlonImage(data=self.im_data, rsc_data=self.rsc_info)
        self.stack_ll = latlon.LatlonImage(data=self.stack_data, rsc_data=self.rsc_info)

    def test_crop_2(self):
        # Should be valid images still
        self.assertTrue(self.im_ll.dem_rsc_is_valid)
        self.assertTrue(self.im_ll[:2, :2].dem_rsc_is_valid)
        self.assertTrue(self.im_ll[:2].dem_rsc_is_valid)

        # Turning into <1D shapes:
        self.assertFalse(self.im_ll[:2, 2].dem_rsc_is_valid)
        self.assertFalse(self.im_ll[2].dem_rsc_is_valid)
        with self.assertRaises(AttributeError):
            self.im_ll[2, 2].dem_rsc_is_valid

    def test_crop_3(self):
        # Should be valid images still
        self.assertTrue(self.stack_ll.dem_rsc_is_valid)
        self.assertTrue(self.stack_ll[:, :2, :2].dem_rsc_is_valid)
        self.assertTrue(self.stack_ll[:2].dem_rsc_is_valid)
        self.assertTrue(self.stack_ll[2].dem_rsc_is_valid)

        # Turning into <1D shapes:
        self.assertFalse(self.stack_ll[:2, 2].dem_rsc_is_valid)
        self.assertFalse(self.stack_ll[2, 2].dem_rsc_is_valid)
        self.assertFalse(self.stack_ll[:, 2, 2].dem_rsc_is_valid)

    def test_contains_floats(self):
        self.assertTrue(latlon.contains_floats(3.4))
        self.assertTrue(latlon.contains_floats((4.5, 3.4)))

        self.assertFalse(latlon.contains_floats(3))
        self.assertFalse(latlon.contains_floats((3, 4)))
        self.assertFalse(latlon.contains_floats((3, None)))
        self.assertFalse(latlon.contains_floats(slice(3, None)))
        self.assertFalse(latlon.contains_floats((1, slice(3, None, None))))

    def test_nearest_pixel(self):
        out2 = self.im_ll.nearest_pixel(lon=-4.6)
        out3 = self.stack_ll.nearest_pixel(lon=-4.6)
        expected = (None, 1)
        self.assertEquals(out2, expected)
        self.assertEquals(out3, expected)

        out2 = self.im_ll.nearest_pixel(lon=-4.6, lat=3.6)
        out3 = self.stack_ll.nearest_pixel(lon=-4.6, lat=3.6)
        expected = (1, 1)
        assert_array_almost_equal(out2, expected)
        assert_array_almost_equal(out3, expected)

        out2 = self.im_ll.nearest_pixel(lon=np.arange(-4.9, -4.6, .1), lat=3.6)
        out3 = self.stack_ll.nearest_pixel(lon=np.arange(-4.9, -4.6, .1), lat=3.6)
        expected = (1, np.array([0, 0, 1, 1]))
        self.assertEquals(out2[0], expected[0])
        self.assertEquals(out3[0], expected[0])
        assert_array_almost_equal(out2[1], expected[1])
        assert_array_almost_equal(out3[1], expected[1])

        out2 = self.im_ll.nearest_pixel(lon=np.arange(-5.3, -4.8, .1), lat=3.6)
        out3 = self.stack_ll.nearest_pixel(lon=np.arange(-5.3, -4.8, .1), lat=3.6)
        expected = (1, np.array([None, 0, 0, 0, 0]))
        assert_array_almost_equal(out2[1][1:], expected[1][1:])
        self.assertEquals(out2[1][0], expected[1][0])
        assert_array_almost_equal(out3[1][1:], expected[1][1:])
        self.assertEquals(out3[1][0], expected[1][0])

    def test_float_indexing(self):
        # Get items by their lat, lon (converted to row, col)
        self.assertEquals(self.im_ll[4.0, -5.0], 0)
        self.assertEquals(self.stack_ll[0, 4.0, -5.0], 0)

        # TODO: fix.. or just stop this in favor of julia
        # assert_array_almost_equal(self.stack_ll[:, 4.0, -5.0], [0, 20, 40])
        # assert_array_almost_equal(self.stack_ll[:2, 4.0, -5.0], [0, 20])

        assert_array_almost_equal(self.im_ll[4.0:3.3, -5.0:-4.0], np.array([[0, 1], [4, 5]]))
        # First two layers, top left 2 box

        assert_array_almost_equal(self.stack_ll[:2, 4.0:3.3, -5.0:-4.0],
                                  np.array([[[0, 1], [4, 5]], [[20, 21], [24, 25]]]))
