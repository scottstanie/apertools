[![Build Status](https://travis-ci.org/scottstanie/apertools.svg?branch=master)](https://travis-ci.org/scottstanie/apertools)

# Apertools: tools for InSAR (Interferometric Synthetic Aperture Radar)


Other helping tools: [sardem](https://github.com/scottstanie/sardem) for creating DEMs from NASA's SRTM, and [sentineleof](https://github.com/scottstanie/sentineleof) for downloading Sentinel 1 EOF precise orbit files.


## Setup and installation

```bash
pip install apertools
```


#### sario.py

Input/Output functions for SAR data.
Contains methods to load Sentinel, UAVSAR and DEM files

Main function: 

```python
import apertools.sario
my_slc = apertools.sario.load('/file/path/radar.slc')
geocoded_slc = apertools.sario.load('/file/path/myslc.geo')
my_int = apertools.sario.load('/file/path/interferogram.int')
unwrapped_int = apertools.sario.load('/file/path/igram.unw')
my_dem = apertools.sario.load('/file/path/elevation.dem')
my_hgt = apertools.sario.load('/file/path/N20W100.hgt')
```

#### latlon.py
Contains LatlonImage class, which loads metadata about an image and acts as a smart numpy array.
Includes functionality for slicing/selecting pixels by latitude/longitude, among other things.

Also contains helper functions for maniuplating lat/lon data.

#### plotting.py
Useful plotting functions, including center-shifted colormap (to make 0 values a neutral color), and 3D stack viewing function


#### los.py
Line of sight utilities


#### gps.py
Several functions for using GPS data in conjunction with InSAR stacks


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
