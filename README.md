[![Build Status](https://travis-ci.org/scottstanie/apertools.svg?branch=master)](https://travis-ci.org/scottstanie/apertools)

# Apertools: tools for InSAR (Interferometric Synthetic Aperture Radar)


Other helping tools: [sentineleof](https://github.com/scottstanie/sentineleof) for downloading Sentinel 1 EOF precise orbit files.


## Setup and installation

```bash
pip install -e apertools
```

There is a version on pypi, but it's pretty outdated. 
I have not tried to keep track of all requirements throughout the entire package, so if you run into `ImportError`s, I have probably installed it using [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) into my existing conda environment.

After installing, the `aper` command line tool is installed, which has multiple sub-commands.

Also installs `asfdownload` for downloading many files from ASF.

The most used modules are...

#### sario.py

Input/Output functions for SAR data.
Contains methods to load Sentinel, UAVSAR and DEM files

Main function: 

```python
from apertools import sario
# Loads images from ROI_PAC format:
my_slc = sario.load('/file/path/radar.slc')
geocoded_slc = sario.load('/file/path/myslc.geo')
my_int = sario.load('/file/path/interferogram.int')
unwrapped_int = sario.load('/file/path/igram.unw')
my_dem = sario.load('/file/path/elevation.dem')

# Can also just pass through to GDAL for any other gdal-readable format.
other_img = sario.load("gdal_readable_image.tif", use_gdal=True)
```

#### gps.py
Functions for using GPS data in conjunction with InSAR stacks.

#### latlon.py
Contains LatlonImage class, which loads metadata about an image and acts as a smart numpy array.
Includes functionality for slicing/selecting pixels by latitude/longitude, among other things.

Also contains helper functions for maniuplating lat/lon data.

#### plotting.py
Useful plotting functions, including center-shifted colormap (to make 0 values a neutral color), and 3D stack viewing function


```python
from apertools import gps
enu_dataframe = gps.load_station_enu("TXKM", start_date="2015-01-01")

from apertools import gps_plots
gps_plots.plot_station_enu("TXKM")
```

#### parsers.py

Classes to deal with extracting relevant data from SAR filenames.
Example:

```python
from apertools.parsers import Sentinel

parser = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70.zip')
parser.start_time
    datetime.datetime(2018, 4, 8, 4, 30, 25)

parser.mission
    'S1A'

parser.polarization
    'DV'
parser.full_parse
('S1A',
 'IW',
 'SLC',
 '_',
 '1',
 'S',
 'DV',
 '20180408T043025',
 '20180408T043053',
 '021371',
 '024C9B',
 '1B70')


parser.field_meanings
('Mission',
 'Beam',
 'Product type',
 'Resolution class',
 'Product level',
 'Product class',
 'Polarization',
 'Start datetime',
 'Stop datetime',
 'Orbit number',
 'data-take identified',
 'product unique id')

```

UAVSAR parser also exists.


#### log.py

Module to make logging pretty with times and module names.

If you also `pip install colorlog`, it will become colored (didn't require this in case people like non-color logs.)

```python
from apertools.log import get_log
logger = get_log()
logger.info("Better than printing")
```

```
[05/29 16:28:19] [INFO log.py] Better than printing
```
