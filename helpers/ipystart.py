get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'tk')

# import apertools
import apertools.sario as sario
import apertools.plotting as plotting
import apertools.utils as utils
import apertools.gps as gps
import apertools.latlon as latlon
# from apertools.scripts import *
import sardem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import glob
import requests
import datetime
import rasterio as rio
