import os


def in_screen():
    """Checks if we are currently running in a screen
    If so, just use agg backend for matplotlib
    (suggestion from https://stackoverflow.com/a/3054314)
    # Answer from: https://stackoverflow.com/a/5392681
    """
    return os.environ.get("STY") is not None


get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

# Need to do this BEFORE pyplot
import matplotlib

if in_screen():
    print("Currently in screen, using agg backend")
    matplotlib.use("agg")
else:
    matplotlib.use('Qt5Agg')

from collections import Counter
import sys
import glob
from pathlib import Path
import requests
import datetime
import itertools
import subprocess

from tqdm import tqdm
from rich import inspect, pretty, print
pretty.install()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import hdf5plugin
# import h5py
import rasterio as rio
# import xarray as xr

plt.ion()

# get_ipython().run_line_magic('matplotlib', 'tk')

# import apertools
import apertools.sario as sario
import apertools.plotting as plotting
import apertools.utils as utils
import apertools.parsers as parsers

# import apertools.gps as gps
# import apertools.latlon as latlon
# from apertools.scripts import *
