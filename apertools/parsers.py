"""
Utilities for parsing file names of SAR products for relevant info.

"""
from __future__ import division, print_function
import os
import re
import pprint
from datetime import datetime

import apertools.latlon
import apertools.utils
from apertools.log import get_log
logger = get_log()

__all__ = ['Sentinel', 'Uavsar', 'SentinelOrbit']


class Base(object):
    """Base parser to illustrate expected interface/ minimum data available
    """
    FILE_REGEX = None
    TIME_FMT = None
    _FIELD_MEANINGS = None

    def __init__(self, filename, verbose=False):
        """
        Extract data from filename
            filename (str): name of SAR/InSAR product
            verbose (bool): print extra logging into about file loading
        """
        self.filename = filename
        self.full_parse()  # Run a parse to check validity of filename
        self.verbose = verbose

    def __str__(self):
        return "{} product: {}".format(self.__class__.__name__, self.filename)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return self.filename < other.filename

    def full_parse(self):
        """Returns all parts of the data contained in filename

        Returns:
            tuple: parsed file data. Entry order will match `field_meanings`

        Raises:
            ValueError: if filename string is invalid
        """
        if not self.FILE_REGEX:
            raise NotImplementedError("Must define class FILE_REGEX to parse")

        match = re.search(self.FILE_REGEX, self.filename)
        if not match:
            raise ValueError('Invalid {} filename: {}'.format(self.__class__.__name__,
                                                              self.filename))
        else:
            return match.groups()

    @property
    def field_meanings(self):
        """List the fields returned by full_parse()"""
        return self._FIELD_MEANINGS

    def _get_field(self, fieldname):
        """Pick a specific field based on its name"""
        idx = self.field_meanings.index(fieldname)
        return self.full_parse()[idx]


class Sentinel(Base):
    """
    Sentinel 1 reference:
    https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions
    or https://sentinel.esa.int/documents/247904/349449/Sentinel-1_Product_Specification

    Example:
        S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70.zip
        S1A_IW_RAW__0SSV_20151018T005110_20151018T005142_008200_00B886_61EC.zip

    File name format:
        MMM_BB_TTTR_LFPP_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_OOOOOO_DDDDDD_CCCC.EEEE

    MMM: mission/satellite S1A or S1B
    BB: Mode/beam identifier. The S1-S6 beams apply to SM products IW,
      EW and WV identifiers appply to products from the respective modes.
    TTT: Product Type: RAW, SLC, GRD, OCN
    R: Resolution class: F, H, M, or _ (N/A)
    L: Processing Level: 0, 1, 2
    F: Product class: S (standard), A (annotation, only used internally)
        - we only care about standard
    PP: Polarization: SH (single HH), SV (single VV), DH (dual HH+HV), DV (dual VV+VH)
    Start date + time (date/time separated by T)
    Stop date + time
    OOOOOO: absolute orbit number: 000001-999999
    DDDDDD: mission data-take identifier: 000001-FFFFFF.
    CCCC: product unique identifier: hexadecimal string from CRC-16 hashing
        the manifest file using CRC-CCITT.

    Once unzipped, the folder extension is always "SAFE"

    Attributes:
        filename (str) name of the sentinel data product
    """
    FILE_REGEX = r'(S1A|S1B)_([\w\d]{2})_([\w_]{3})([FHM_])_([012])S([SDHV]{2})_([T\d]{15})_([T\d]{15})_(\d{6})_([\d\w]{6})_([\d\w]{4})'
    TIME_FMT = '%Y%m%dT%H%M%S'
    _FIELD_MEANINGS = (
        'mission',
        'beam',
        'product type',
        'resolution class',
        'product level',
        'polarization',
        'start datetime',
        'stop datetime',
        'orbit number',
        'data-take identified',
        'product unique id',
    )

    def __init__(self, filename, **kwargs):
        super(Sentinel, self).__init__(filename, **kwargs)
        # The name of the unzipped .SAFE directory (with .zip stripped)
        self._safe_dir = self._form_safe_dir(filename)
        self._preview_folder = os.path.join(self._safe_dir, 'preview')
        self.map_overlay_kml = os.path.join(self._preview_folder, 'map-overlay.kml')
        self._lon_lat_overlay_coords = apertools.latlon.map_overlay_coords(self.map_overlay_kml)


# TODO: in 30_20181107T092858_024480_02AF1F_BD80.iso.xml
#                    <gmd:EX_BoundingPolygon>
#                      <gmd:polygon>
#                        <gml:Polygon srsName="http://www.opengis.net/gml/srs/epsg.xml#4326" srsDimension="2" gml:id="boundingPolygon">
#                          <gml:exterior>
#                            <gml:LinearRing>
#                              <gml:pos>64.329208 -42.509876</gml:pos>
#                              <gml:pos>64.784164 -47.771400</gml:pos>
#                              <gml:pos>66.444023 -47.122295</gml:pos>
#                              <gml:pos>65.972809 -41.522572</gml:pos>
#                              <gml:pos>64.329208 -42.509876</gml:pos>
#                            </gml:LinearRing>
#                          </gml:exterior>
#                        </gml:Polygon>
#                      </gmd:polygon>
#                    </gmd:EX_BoundingPolygon>
# or
#                      <gmd:eastBoundLongitude>
#                        <gco:Decimal>-41.522572</gco:Decimal>
#                      </gmd:eastBoundLongitude>
#                      <gmd:southBoundLatitude>
#                        <gco:Decimal>64.329208</gco:Decimal>
#                      </gmd:southBoundLatitude>

    def _form_safe_dir(self, filename):
        """Get just the Sentinel product name without extensions, then add .SAFE"""
        # Strip '/' from end to start in case they pass "blahblah.SAFE/", or splitext[1] is ''
        fname = filename.rstrip('/').replace('.zip', '').replace('.geo', '')
        root, ext = os.path.splitext(fname)
        return root + '.SAFE'

    def __str__(self):
        return "{} {}, path {} from {}".format(self.__class__.__name__, self.mission, self.path,
                                               self.date)

    def __lt__(self, other):
        return (self.start_time, self.filename) < (other.start_time, other.filename)

    def __eq__(self, other):
        # TODO: Do we just want to compare product_uids?? or filenames?
        return self.product_uid == other.product_uid
        # return self.filename == other.filename

    def __hash__(self):
        return hash(self.product_uid)

    @property
    def start_time(self):
        """Returns start datetime from a sentinel file name

        Args:
            filename (str): filename of a sentinel 1 product

        Returns:
            datetime: start datetime of mission


        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.start_time)
            2018-04-08 04:30:25
        """
        start_time_str = self._get_field('start datetime')
        return datetime.strptime(start_time_str, self.TIME_FMT)

    @property
    def stop_time(self):
        """Returns stop datetime from a sentinel file name

        Args:
            filename (str): filename of a sentinel 1 product

        Returns:
            datetime: stop datetime

        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.stop_time)
            2018-04-08 04:30:53
        """
        stop_time_str = self._get_field('stop datetime')
        return datetime.strptime(stop_time_str, self.TIME_FMT)

    @property
    def polarization(self):
        """Returns type of polarization of product

        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.polarization)
            DV
        """
        return self._get_field('polarization')

    @property
    def product_type(self):
        """Returns product type/level

        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.product_type)
            SLC
        """
        return self._get_field('product type')

    @property
    def level(self):
        """Alias for product type/level """
        return self.product_type

    @property
    def mission(self):
        """Returns satellite/mission of product (S1A/S1B)

        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.mission)
            S1A
        """
        return self._get_field('mission')

    @property
    def absolute_orbit(self):
        """Absolute orbit of data, included in file name

        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.absolute_orbit)
            21371
        """
        return int(self._get_field('orbit number'))

    @property
    def relative_orbit(self):
        """Relative orbit number/ path

        Formulas for relative orbit from absolute come from:
        https://forum.step.esa.int/t/sentinel-1-relative-orbit-from-filename/7042

        Example:
            >>> s = Sentinel('S1A_IW_SLC__1SDV_20180408T043025_20180408T043053_021371_024C9B_1B70')
            >>> print(s.relative_orbit)
            124
            >>> s = Sentinel('S1B_WV_OCN__2SSV_20180522T161319_20180522T164846_011036_014389_67D8')
            >>> print(s.relative_orbit)
            160
        """
        if self.mission == 'S1A':
            return ((self.absolute_orbit - 73) % 175) + 1
        elif self.mission == 'S1B':
            return ((self.absolute_orbit - 27) % 175) + 1

    @property
    def path(self):
        """Alias for relative orbit number"""
        return self.relative_orbit

    @property
    def product_uid(self):
        """Unique identifier of product (last 4 of filename)"""
        return self._get_field('product unique id')

    @property
    def date(self):
        """Date of acquisition: shortcut for start_time.date()"""
        return self.start_time.date()

    @property
    def swath_extent(self):
        """Give the lon and lat boundaries for the swath

        Matches latlon.grid_extent(**dem_rsc_data)
        (lon_left,lon_right,lat_bottom,lat_top)
        """
        if not os.path.exists(self.map_overlay_kml):
            raise ValueError("No map-overlay.kml file found in %s" % self._preview_folder)
        lons, lats = list(zip(*self._lon_lat_overlay_coords))
        return min(lons), max(lons), min(lats), max(lats)

    @property
    def extent(self):
        """alias for swath_extent"""
        return self.swath_extent

    @property
    def swath_width_height(self):
        """Width and height of the swath area in degrees"""
        left, right, bot, top = self.swath_extent
        return right - left, top - bot

    def overlaps_dem(self, dem_rsc_data):
        """Swath is contained in DEM from rsc data"""
        dem_extent = apertools.latlon.grid_extent(**dem_rsc_data)
        return apertools.latlon.intersects(self.swath_extent, dem_extent)

    def overlaps_geojson(self, geojson):
        """Swath is contained in bounding box for geojson"""
        if isinstance(geojson, str):
            geojson = apertools.sario.load(geojson)

        geojson_extent = apertools.geojson.extent(geojson)
        return apertools.latlon.intersects(self.swath_extent, geojson_extent)

    def overlaps(self, geojson_or_rsc):
        """Swath is contained in DEM or geojson (parsed to figure out which is passed)"""
        if 'x_first' in [k.lower() for k in geojson_or_rsc.keys()]:
            return self.overlaps_dem(geojson_or_rsc)

        # Test for geojson by extracting the coords
        try:
            apertools.geojson.coords(geojson_or_rsc)
        except ValueError:
            raise ValueError("Need either valid geojson or dem_rsc_data")
        return self.overlaps_geojson(geojson_or_rsc)


class SentinelOrbit(Base):
    """
    Sentinel 1 orbit reference:
    https://sentinel.esa.int/documents/247904/351187/GMES_Sentinels_POD_Service_File_Format_Specification
        section 2
    https://qc.sentinel1.eo.esa.int/doc/api/
    https://sentinels.copernicus.eu/documents/247904/3372484/Copernicus-POD-Regular-Service-Review-Jun-Sep-2018.pdf
        see here (section 3.6) for differences in orbit accuracy)

    Example:
        S1A_OPER_AUX_PREORB_OPOD_20200325T131800_V20200325T121452_20200325T184952.EOF

    The filename must comply with the following pattern:
        MMM_CCCC_TTTTTTTTTT_<instance_id>.EOF

    MMM = mission, S1A or S1B
    CCCC =  File Class, we only want OPER = routine operational
    TTTTTTTTTT = File type
     = FFFF DDDDDD
        FFFF = file category, we want AUX_:auxiliary data files;
        DDDDDD = Semantic Descriptor
        most common = POEORB: Precise Orbit Ephemerides (POE) Orbit File
            (available after 1-2 weeks)
        also, RESORB: Restituted orbit file
            (covers 6 hour windows, less accurate, more immediate)
        TODO: do I ever want to deal with the AUX antenna files?

    <instance id> has a couple:
    ssss_yyyymmddThhmmsswhere:
        ssss is the Site Centre of the file originator (OPOD for S-1 and S-2)
        and a validity start/stop, same date format

    Attributes:
        filename (str) name of the sentinel data product
    """
    FILE_REGEX = r'(S1A|S1B)_OPER_AUX_([\w_]{6})_OPOD_([T\d]{15})_V([T\d]{15})_([T\d]{15}).EOF'
    TIME_FMT = '%Y%m%dT%H%M%S'
    _FIELD_MEANINGS = ('mission', 'orbit type', 'created datetime', 'start datetime',
                       'stop datetime')

    def __init__(self, filename, **kwargs):
        super(SentinelOrbit, self).__init__(filename, **kwargs)

    def __str__(self):
        return "{} {} from {} to {}".format(self.orbit_type, self.__class__.__name__,
                                            self.start_time, self.stop_time)

    def __lt__(self, other):
        return (self.start_time, self.filename) < (other.start_time, other.filename)

    def __contains__(self, dt):
        """Checks if a datetime lies within the validity window"""
        return self.start_time < dt < self.stop_time

    def __eq__(self, other):
        return (
            self.mission,
            self.start_time,
            self.stop_time,
            self.orbit_type,
        ) == (
            other.mission,
            other.start_time,
            other.stop_time,
            other.orbit_type,
        )

    @property
    def mission(self):
        """Returns satellite/mission of product (S1A/S1B)

        Example:
            >>> s = SentinelOrbit('S1A_OPER_AUX_POEORB_OPOD_20200121T120654_V20191231T225942_20200102T005942.EOF')
            >>> print(s.mission)
            S1A
        """
        return self._get_field('mission')

    @property
    def start_time(self):
        """Returns start datetime of an orbit

        Args:
            filename (str): filename of a sentinel 1 EOF file

        Returns:
            datetime: start datetime of mission

        Example:
            >>> s = SentinelOrbit('S1A_OPER_AUX_POEORB_OPOD_20200121T120654_V20191231T225942_20200102T005942.EOF')
            >>> print(s.start_time)
            2019-12-31 22:59:42
        """
        start_time_str = self._get_field('start datetime')
        return datetime.strptime(start_time_str, self.TIME_FMT)

    @property
    def stop_time(self):
        """Returns stop datetime from a sentinel file name

        Args:
            filename (str): filename of a sentinel 1 product

        Returns:
            datetime: stop datetime

        Example:
            >>> s = SentinelOrbit('S1A_OPER_AUX_POEORB_OPOD_20200121T120654_V20191231T225942_20200102T005942.EOF')
            >>> print(s.stop_time)
            2020-01-02 00:59:42
        """
        stop_time_str = self._get_field('stop datetime')
        return datetime.strptime(stop_time_str, self.TIME_FMT)

    @property
    def created_time(self):
        """Returns created datetime from a orbit file name

        Args:
            filename (str): filename of a sentinel 1 product

        Returns:
            datetime: created datetime

        Example:
            >>> s = SentinelOrbit('S1A_OPER_AUX_POEORB_OPOD_20200121T120654_V20191231T225942_20200102T005942.EOF')
            >>> print(s.created_time)
            2020-01-21 12:06:54
        """
        stop_time_str = self._get_field('created datetime')
        return datetime.strptime(stop_time_str, self.TIME_FMT)

    @property
    def orbit_type(self):
        """Type of orbit file (previse, restituted)

        Example:
        >>> s = SentinelOrbit('S1A_OPER_AUX_POEORB_OPOD_20200121T120654_V20191231T225942_20200102T005942.EOF')
        >>> print(s.orbit_type)
        precise
        >>> s = SentinelOrbit('S1B_OPER_AUX_RESORB_OPOD_20200325T151938_V20200325T112442_20200325T144212.EOF')
        >>> print(s.orbit_type)
        restituted
        """
        o = self._get_field('orbit type')
        if o == 'POEORB':
            return 'precise'
        elif o == 'RESORB':
            return 'restituted'
        elif o == 'PREORB':
            return 'predicted'
        else:
            raise ValueError("unknown orbit type: %s" % self.filename)

    @property
    def date(self):
        """Date of acquisition: shortcut for start_time.date()"""
        return self.start_time.date()


class Uavsar(Base):
    """Uavsar reference for Polsar:
    https://uavsar.jpl.nasa.gov/science/documents/polsar-format.html

    RPI/ InSAR format reference:
    https://uavsar.jpl.nasa.gov/science/documents/rpi-format-browse.html

    Naming example:
    Dthvly_34501_08038_006_080731_L090HH_XX_01.slc

    Dthvly is the site name, 345 degrees is the heading of UAVSAR in flight,
    with a counter of 01, the flight was the thirty-eighth flight by UAVSAR in
    2008,this data take was the sixth data take during the flight, the data was
    acquired on July 31, 2008 (UTC), the frequency band was L-band, pointing at
    perpendicular to the flight heading (90 degrees counterclockwise), this
    file contains the HH data, this is the first interation of processing,
    cross talk calibration has not been applied, and the data type is SLC.

    For downsampled products (3x3 and 5x5), there is an optional extension
    of _ML3X3 and _ML5X5 tacked onto the end

    Examples:
        >>> fname = 'Dthvly_34501_08038_006_080731_L090HH_XX_01.slc'
        >>> parser = Uavsar(fname)

    """
    FILE_REGEX = r'([\w\d]{6})_(\d{3})(\w+)_(\d{2})(\d{3})_(\d{3})_(\d{6})_(\w)(\d{3})(\w{0,4})_(XX|CX)_(\w{2})(_ML\dX\d)?'
    TIME_FMT = '%y%m%d'
    _FIELD_MEANINGS = (
        'target site',
        'heading',
        'counter',
        'flight year',
        'flight number',
        'flight line',
        'date',
        'frequency band',
        'steering angle',
        'polarization',
        'cross talk',
        'version number',
        'downsampling',
    )
    # Filetype of real or complex depends on the polarization for .grd, .mlc
    REAL_POLS = ('HHHH', 'HVHV', 'VVVV')
    COMPLEX_POLS = ('HHHV', 'HHVV', 'HVVV')
    POLARIZATIONS = REAL_POLS + COMPLEX_POLS

    def __str__(self):
        return "{} {} from {}".format(self.__class__.__name__, self.polarization, self.date)

    @property
    def date(self):
        """Returns date of flight from file name
        Args:
            filename (str): filename of a product from self

        Returns:
            datetime.date: date mission

        Examples:
            >>> parser = Uavsar('Dthvly_34501_08038_006_080731_L090HH_XX_01.slc')
            >>> parser.date
            datetime.date(2008, 7, 31)
        """
        date_str = self._get_field('date')
        return datetime.strptime(date_str, self.TIME_FMT).date()

    @property
    def polarization(self):
        """Polarization of the product, if any

        May be between 2 and 4 chars, though .zip files don't have one

        Examples:
            >>> Uavsar('brazos_14938_17087_004_170831_L090HH_CX_01.slc').polarization
            'HH'
            >>> Uavsar('brazos_14938_17087_004_170831_L090HHHV_CX_01.mlc').polarization
            'HHHV'
            >>> Uavsar('brazos_14938_17087_004_170831_L090_CX_01_grd.zip').polarization
            ''
        """
        return self._get_field('polarization')

    @property
    def target_site(self):
        """Target site of acquisition"""
        return self._get_field('target site')

    @property
    def downsampling(self):
        """Amount of downsampling of product, if any

        Examples:
        >>> print(Uavsar('brazos_14938_17087_004_170831_L090_CX_01_grd.zip').downsampling)
        None
        >>> Uavsar('brazos_14938_17087_004_170831_L090_CX_01_ML5X5_grd.zip').downsampling
        '5X5'
        >>> Uavsar('brazos_14938_17087_004_170831_L090HHHV_CX_01_ML3X3.grd').downsampling
        '3X3'
        """
        sample_str = self._get_field('downsampling')
        return sample_str.replace('_ML', '') if sample_str else None

    def _make_ann_filename(self):
        """Take the name of a data file and return corresponding .ann name

        Examples:
        >>> u = Uavsar('brazos_14938_17087_004_170831_L090HHHV_CX_01_ML3X3.grd')
        >>> print(u.ann_filename)
        brazos_14938_17087_004_170831_L090_CX_01_ML3X3.ann
        >>> u = Uavsar('brazos_14938_17087_004_170831_L090_CX_01.int')
        >>> print(u.ann_filename)
        brazos_14938_17087_004_170831_L090_CX_01.ann
        """

        # The .mlc and .grd files have polarization added to filename, .ann files don't
        shortname = self.filename
        for p in self.POLARIZATIONS:
            shortname = shortname.replace(p, '')

        ext = apertools.utils.get_file_ext(shortname)
        # If this is a block we split up and names .1.int, remove that since
        # all have the same .ann file
        shortname = re.sub('\.\d' + ext, ext, shortname)
        if ext == ".grd":
            # .int.grd is full ext
            full_ext = '.'.join(shortname.split('.')[1:])
            return shortname.replace(full_ext, 'ann')

        return shortname.replace(ext, '.ann')

    @property
    def ann_filename(self):
        """The name of the corresponding .ann file"""
        return self._make_ann_filename()

    @property
    def ann_data(self):
        """The dict of ann data for a file

        Note: This will try to read the file, so it must exist
        (i.e. can't just pass a valid string filename without a file)
        """
        return self.parse_ann_file()

    def parse_ann_file(self):
        return parse_ann_file(self.ann_filename, filename=self.filename, verbose=self.verbose)


class UavsarInt(Uavsar):
    """See https://uavsar.jpl.nasa.gov/science/documents/rpi-format.html"""
    FILE_REGEX = r'([\w\d]{6})_(\d{2})(\d{3})_(\d{2})(\d{3})-(\d{3})_(\d{5})-(\d{3})_(\d{4})d_s01_([\w\d]{6,8})_(\d{2})'
    _FIELD_MEANINGS = (
        'target site',
        'heading',
        'counter',
        'track1 year',
        'track1 flight number',
        'track1 flight line',
        # TODO: fill in rest
        'flight year',
        'flight number',
        'flight line',
    )

    def __str__(self):
        return "{} from {}".format(self.__class__.__name__, self.target_site)


def parse_ann_file(ann_filename, filename=None, ext=None, verbose=False):
    """Returns the requested data from the UAVSAR annotation in ann_filename

    Args:
        ann_data (dict): key-values of requested data from .ann file

    Returns:
        dict: the annotation file parsed into a dict. If no annotation file
            can be found, None is returned
    """
    def _parse_line(line):
        wordlist = line.split()
        # Pick the entry after the equal sign when splitting the line
        return wordlist[wordlist.index('=') + 1]

    def _parse_int(line):
        return int(_parse_line(line))

    def _parse_float(line):
        return float(_parse_line(line))

    def _make_line_regex(ext, field):
        return r'{}.{}'.format(line_keywords.get(ext), field)

    if not ext:
        if not filename:
            raise ValueError("Need either filename or ext")
        ext = apertools.utils.get_file_ext(filename)

    if verbose:
        logger.info("Trying to load ann_data from %s", ann_filename)
    if not os.path.exists(ann_filename):
        if verbose:
            logger.info("No file found: returning None")
        return None

    # Taken from a .ann file: (need to check if this is always true?)
    # SLC Data Units = linear amplitude
    # MLC Data Units = linear power
    # GRD Data Units = linear power
    ann_data = {}
    line_keywords = {
        # ext: line start term
        '.slc': 'slc_mag',
        '.mlc': 'mlc_mag',
        '.int': 'slt',
        '.unw': 'slt',
        '.cor': 'slt',
        '.amp': 'slt',
        '.grd': 'grd_mag',
    }
    # Add extra .grd extensions
    for e in ('.int', '.unw', '.cc', '.cor', '.amp1', '.amp2'):
        line_keywords[e + '.grd'] = line_keywords['.grd']

    row_key = line_keywords.get(ext) + '.set_rows'
    col_key = line_keywords.get(ext) + '.set_cols'

    # Peg position the nadir position of aircraft at middle of datatake
    with open(ann_filename, 'r') as f:
        for line in f.readlines():
            if line.startswith(row_key):
                ann_data['rows'] = _parse_int(line)
                # Also add .rsc equivalent for compatibility
                ann_data['file_length'] = ann_data['rows']
            elif line.startswith(col_key):
                ann_data['cols'] = _parse_int(line)
                ann_data['width'] = ann_data['cols']
            # Center Latitude of Upper Left Pixel of GRD image, or
            # range Offset(R0) from Peg in meters
            # Note: using convention of .rsc files for consitency
            # I.E. x_first, x_step, y_first, y_step
            elif re.match(_make_line_regex(ext, 'row_addr'), line):
                ann_data['y_first'] = _parse_float(line)
            # Center Longitude of Upper Left Pixel
            elif re.match(_make_line_regex(ext, 'col_addr'), line):
                ann_data['x_first'] = _parse_float(line)
            # GRD Latitude Pixel Spacing
            # the step is negative in the y (row) direction
            elif re.match(_make_line_regex(ext, 'row_mult'), line):
                ann_data['y_step'] = _parse_float(line)
            # GRD Longitude Pixel Spacing or SLC R (range) Slant Post Spacing
            elif re.match(_make_line_regex(ext, 'col_mult'), line):
                ann_data['x_step'] = _parse_float(line)
            # TODO: Add more parsing! whatever is useful from .ann file

    if verbose:
        logger.info(pprint.pformat(ann_data))

    return ann_data
