import unittest
from datetime import datetime
from os.path import join, dirname

from apertools.parsers import Sentinel, Uavsar


class TestSentinel(unittest.TestCase):
    def setUp(self):
        self.filename = 'S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70.zip'
        self.parser = Sentinel(self.filename)

    def test_bad_filename(self):
        self.assertRaises(ValueError, Sentinel, 'asdf')
        self.assertRaises(ValueError, Sentinel, 'A_b_c_d_e_f_g_h_i_j_k_l')

    def test_full_parse(self):
        expected_output = ('S1A', 'IW', 'SLC', '_', '1', 'DV', '20180408T043025', '20180408T043053',
                           '021371', '024C9B', '1B70')

        self.assertEqual(self.parser.full_parse(), expected_output)

    def test_path_parse(self):
        path_filename = '/some/path/' + self.filename
        self.assertEqual(Sentinel(path_filename).full_parse(), self.parser.full_parse())

    def test_start_time(self):
        expected_start = datetime(2018, 4, 8, 4, 30, 25)
        self.assertEqual(self.parser.start_time, expected_start)

    def test_stop_time(self):
        expected_stop = datetime(2018, 4, 8, 4, 30, 53)
        self.assertEqual(self.parser.stop_time, expected_stop)

    def test_polarization(self):
        self.assertEqual(self.parser.polarization, 'DV')

    def test_mission(self):
        self.assertEqual(self.parser.mission, 'S1A')


class TestUavsar(unittest.TestCase):
    def setUp(self):
        self.datapath = join(dirname(__file__), 'data')
        self.ann = join(self.datapath, 'brazos_14937_17090_017_170903_L090_CX_01.ann')
        self.int = self.ann.replace('.ann', '.int')
        self.grd = self.ann.replace('.ann', '.grd')
        self.slc = self.ann.replace('.ann', '.slc')
        self.parser_int = Uavsar(self.int, verbose=True)
        self.parser_grd = Uavsar(self.grd)
        self.parser_slc = Uavsar(self.slc)

    def test_parse_ann_file(self):
        int_ann_info = self.parser_int.parse_ann_file()
        expected_ann_info = {
            'cols': 3300,
            'width': 3300,
            'rows': 22826,
            'file_length': 22826,
            'x_first': 13450.19161366,
            'x_step': 4.99654098,
            'y_first': -84242.1,
            'y_step': 7.2
        }
        self.assertEqual(expected_ann_info, int_ann_info)

        # Same path and same name as .ann file
        # Different data for the .slc for same ann
        expected_ann_info = {
            'cols': 9900,
            'width': 9900,
            'rows': 273921,
            'file_length': 273921,
            'x_first': 13448.5261,
            'x_step': 1.66551366,
            'y_first': -84245.4,
            'y_step': 0.6
        }

        slc_ann_info = self.parser_slc.parse_ann_file()
        self.assertEqual(expected_ann_info, slc_ann_info)

        expected_ann_info = {
            'cols': 19322,
            'width': 19322,
            'rows': 25751,
            'file_length': 25751,
            'x_first': -96.2685342,
            'x_step': 5.556e-05,
            'y_first': 30.279311040000003,
            'y_step': -5.556e-05
        }
        grd_ann_info = self.parser_grd.parse_ann_file()
        self.assertEqual(expected_ann_info, grd_ann_info)
