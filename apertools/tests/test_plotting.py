import unittest
from os.path import join, dirname
import tempfile
import shutil
import matplotlib.pyplot as plt

from insar import plotting, timeseries, sario


class TestPlotting(unittest.TestCase):
    def setUp(self):
        self.datapath = join(dirname(__file__), 'data')
        igram_path = join(self.datapath, 'sbas_test')
        self.igram_path = igram_path
        self.stack = sario.load_stack(directory=igram_path, file_ext=".unw")

    def test_animate_stack(self):
        try:
            # Commands to turn off interactive for travis tests
            plt.ioff()

            temp_dir = tempfile.mkdtemp()
            plotting.animate_stack(self.stack, display=False)

            igram_files = timeseries.read_intlist(join(self.igram_path, 'intlist'), parse=False)
            plotting.animate_stack(self.stack, display=False, titles=igram_files)

        finally:
            shutil.rmtree(temp_dir)
