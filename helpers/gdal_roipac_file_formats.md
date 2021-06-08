
# Reference on BSQ, BIL, BIP

https://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/bil-bip-and-bsq-raster-files.htm

Note: Our files are only BIL (when we have, e.g., amplitude interleaved with correlation), or 1 band complex (where format doesn't matter).
BUT if you consider the "real" and "imaginary" as two separate bands, then the interferograms (which are really 1 band complex), would be in "BIP",
since it starts with both bands' data for pixel (1,1), then both bands for (1,2),... etc.

[Check here for another image of BIP/BIL/BSQ](https://bytebucket.org/hu-geomatics/enmap-box-idl/wiki/img/migrated/2509213088-image001.gif?rev=786971c259c9bdb49669f8fb2f66ccdee6cefe20)


## Note on loading ROI PAC files in with gdal
If you have a .rsc file with the same base name as the data file, it will automatically load.

E.G.
```python
$ cp dem.rsc 20180205_20180217.amp.rsc
$ ls 20180205_20180217*
20180205_20180217.amp     20180205_20180217.amp.rsc  ...

$ python
>>> from osgeo import gdal
>>> f1 = gdal.Open("20180205_20180217.amp")

>>> f1.GetDriver().LongName
'ROI_PAC raster'

>>> f1.RasterCount
2

>>> band1 = f1.GetRasterBand(1)

>>> band1.ReadAsArray()[:4, :4]
array([[168.60542, 182.9884 , 136.90907, 149.62862],
       [162.02464, 187.2191 , 146.78882, 124.4787 ],
       [148.24612, 149.8064 , 170.14888, 158.08841],
       [151.24222, 174.0751 , 191.23549, 162.8879 ]], dtype=float32)
```

# Info on filetypes
|                                  |         extension (ROIPAC/zebker)        | Data type (numpy/Gdal) |                         File type                        |
|:--------------------------------:|:----------------------------------------:|:----------------------:|:--------------------------------------------------------:|
|  SAR image (SLC, or geocoded SLC |  (.slc, regular SAR/ .geo, geocoded SAR) |   Complex64/CFloat32   |  1 band (or BIP for 2 bands  of real floats, imag float) |
|          Interferograms          |                   .int                   |   Complex64/CFloat32   |                      1 complex band                      |
|          Unwrapped igram         |                   .unw                   |     Float32/Float32    |                        2 band BIL                        |
|            correlation           |               (.cor / .cc)               |         Float32        |                        2 band BIL                        |
|             amplitude            |                   .amp                   |         Float32        |              2 band BIP (bands = SAR images)             |

# VRT Files (todo)
