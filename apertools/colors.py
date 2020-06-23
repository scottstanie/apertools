from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

def make_dismph_colormap():
    """Make a custom colormap like the one used in dismph.

    The list was created from dismphN.mat in geodmod which is a 64 segmented colormap
    using the following:
      from scipy.io import loadmat
      cmap = loadmat('dismphN.mat',struct_as_record=True)['dismphN']
      from matplotlib.colors import rgb2hex
      list=[]
      for i in cmap: list.append(rgb2hex(i))

    Source: https://imaging.unavco.org/data/geoslc/geoslc2intf.py
    """
    colors = [
        '#f579cd', '#f67fc6', '#f686bf', '#f68cb9', '#f692b3', '#f698ad', '#f69ea7', '#f6a5a1',
        '#f6ab9a', '#f6b194', '#f6b78e', '#f6bd88', '#f6c482', '#f6ca7b', '#f6d075', '#f6d66f',
        '#f6dc69', '#f6e363', '#efe765', '#e5eb6b', '#dbf071', '#d0f477', '#c8f67d', '#c2f684',
        '#bbf68a', '#b5f690', '#aff696', '#a9f69c', '#a3f6a3', '#9cf6a9', '#96f6af', '#90f6b5',
        '#8af6bb', '#84f6c2', '#7df6c8', '#77f6ce', '#71f6d4', '#6bf6da', '#65f6e0', '#5ef6e7',
        '#58f0ed', '#52e8f3', '#4cdbf9', '#7bccf6', '#82c4f6', '#88bdf6', '#8eb7f6', '#94b1f6',
        '#9aabf6', '#a1a5f6', '#a79ef6', '#ad98f6', '#b392f6', '#b98cf6', '#bf86f6', '#c67ff6',
        '#cc79f6', '#d273f6', '#d86df6', '#de67f6', '#e561f6', '#e967ec', '#ed6de2', '#f173d7'
    ]
    dismphCM = LinearSegmentedColormap.from_list('dismph', colors)
    dismphCM.set_bad('w', 0.0)
    return dismphCM


def discrete_seismic_colors(n=5):
    """From http://colorbrewer2.org/#type=diverging&scheme=RdBu&n=7"""
    if n == 5:
        return list(
            np.array([
                (5, 113, 176, 256),
                (146, 197, 222, 256),
                # (247, 247, 247, 245),  # To make red-white-blue
                (247, 247, 191, 245),  # To make red-yellow-blue
                (244, 165, 130, 256),
                (202, 0, 32, 256),
            ]) / 256)
    elif n == 7:
        # Really this is red-yellow-blue
        # http://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=7
        return list(
            np.array([
                (69, 117, 199, 256),
                (145, 191, 219, 256),
                (224, 243, 248, 256),
                (255, 255, 191, 255),
                (254, 224, 144, 256),
                (252, 141, 89, 256),
                (215, 48, 39, 256),
            ]) / 256)
        # Using the Red-white-blue
        # return list(
        #     np.array([
        #         (33, 102, 172, 256),
        #         (103, 169, 207, 256),
        #         (209, 229, 240, 256),
        #         (247, 247, 247, 250),
        #         (253, 219, 199, 256),
        #         (239, 138, 98, 256),
        #         (178, 24, 43, 256),
        #     ]) / 256)


SEISMIC_WIDE2 = LinearSegmentedColormap.from_list(
    "seismic_wider",
    [
        (0, 0, .3, 1),
        (0, 0, .7, 1),
        (.1, .1, .9, 1),
        (.3, .3, .95, 1),
        (.6, .6, 1, 1),
        (.85, .85, 1, 1),
        (.92, .92, 1, .99),
        (.98, .98, 1, .98),
        (1, 1, 1, .95),
        (1, .98, .98, .98),
        (1, .92, .92, .99),
        (1, .85, .85, 1),
        (1, 0.6, 0.6, 1),
        (1, 0.3, 0.3, 1),
        (.9, .1, .1, 1),
        (.7, 0, 0, 1),
        (.3, 0, 0, 1),
    ],
    N=250,
)
plt.register_cmap("seismic_wider", SEISMIC_WIDE2)
SEISMIC_WIDE_Y = LinearSegmentedColormap.from_list(
    "seismic_wide_y",
    [
        (0, 0, .7, 1),
        (0, 0, 1, 1),
        (.1, .3, 1, 1),
        (.2, .6, .95, 1),
        (.5, .8, .85, 1),
        (.8, .95, .80, 1),
        (.9, .92, .78, 1),
        (.95, .95, .75, 1),
        (.95, .92, .70, 1),
        (.95, .85, .65, 1),
        (.97, .65, .3, 1),
        (1, 0.3, 0.2, 1),
        (1, 0.1, 0.05, 1),
        (1, 0, 0, 1),
        (.7, 0, 0, 1),
    ],
    N=450,
)
plt.register_cmap("seismic_wide_y", SEISMIC_WIDE_Y)

DISMPH = make_dismph_colormap()
plt.register_cmap(cmap=DISMPH)
DISCRETE_SEISMIC5 = LinearSegmentedColormap.from_list('discrete_seismic5',
                                                      discrete_seismic_colors(5),
                                                      N=5)
plt.register_cmap(cmap=DISCRETE_SEISMIC5)
DISCRETE_SEISMIC7 = LinearSegmentedColormap.from_list('discrete_seismic7',
                                                      discrete_seismic_colors(7),
                                                      N=7)
plt.register_cmap(cmap=DISCRETE_SEISMIC7)

SEISMIC_Y = LinearSegmentedColormap.from_list('seismic_y', discrete_seismic_colors(7), N=250)
plt.register_cmap(cmap=SEISMIC_Y)
SEISMIC_Y2 = LinearSegmentedColormap.from_list('seismic_y2', discrete_seismic_colors(5), N=250)
plt.register_cmap(cmap=SEISMIC_Y2)
SEISMIC_WIDE = LinearSegmentedColormap.from_list(
    'seismic_wide',
    [(0, 0, .3, 1), (0, 0, 1, 1), (.6, .6, 1, 1), (.9, .9, 1, 1), (1, 1, 1, 1), (1, .9, .9, 1),
     (1, 0.6, 0.6, 1), (1, 0, 0, 1), (.5, 0, 0, 1)],  # Extra white in middle from seismic
    N=250,
)
plt.register_cmap(cmap=SEISMIC_WIDE)


def shifted_color_map(cmap, start=0, midpoint=0.5, stop=1.0, num_levels=None):
    """Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Attribution: https://stackoverflow.com/a/20528097, Paul H

    Args:
      cmap (str or matplotlib.cmap): The matplotlib colormap to be altered.
          Can be matplitlib.cm.seismic or 'seismic'
      start (float): Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint (float): The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop (float): Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
      num_levels (int): form fewer discrete levels in the colormap

    Returns:
        matplotlib.cmap
    """
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap, num_levels)

    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

    # regular index to compute the colors
    # N = num_levels
    N = 256
    reg_index = np.linspace(start, stop, N + 1)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, N // 2, endpoint=False),
        np.linspace(midpoint, 1.0, N // 2 + 1, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = LinearSegmentedColormap('shiftedcmap', cdict, N=num_levels or N)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def make_shifted_cmap(img=None, vmax=None, vmin=None, cmap_name='seismic', num_levels=None):
    """Scales the colorbar so that 0 is always centered (white)"""
    if img is not None:
        if vmin is None:
            vmin = np.nanmin(img)
        if vmax is None:
            vmax = np.nanmax(img)

    if vmax is None or vmin is None:
        raise ValueError("Required args: img, or vmax and vmin")
    midpoint = 1 - vmax / (abs(vmin) + vmax)
    return shifted_color_map(cmap_name, midpoint=midpoint, num_levels=num_levels)


def cmap_to_qgis(cmap_rgba_arr):
    # TODO: add the xml stuff
    stops = np.linspace(1 / 257, 256 / 257, len(cmap_rgba_arr))
    rows = []
    for s, rgb in zip(stops, cmap_rgba_arr):
        ss = "%.6f" % s
        rows.append(ss + ';' + ','.join(rgb.astype(int)))
    return ':'.join(rows)


# <!DOCTYPE qgis_style>
# <qgis_style version="1">
#   <symbols/>
#   <colorramps>
#     <colorramp type="gradient" name="seismic_wide_y" favorite="1">
#       <prop k="color1" v="0,0,178,255"/>
#       <prop k="color2" v="178,0,0,255"/>
#       <prop k="discrete" v="0"/>
#       <prop k="rampType" v="gradient"/>
#       <prop k="stops" v="0.003891;0,0,178,255:0.074764;0,..."/>
#     </colorramp>
#   </colorramps>
# </qgis_style>
