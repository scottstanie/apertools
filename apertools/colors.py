from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


MATLAB_COLORS = [
    [0, 0.4470, 0.7410, 1],
    [0.8500, 0.3250, 0.0980, 1],
    [0.9290, 0.6940, 0.1250, 1],
    [0.4940, 0.1840, 0.5560, 1],
    [0.4660, 0.6740, 0.1880, 1],
    [0.3010, 0.7450, 0.9330, 1],
    [0.6350, 0.0780, 0.1840, 1],
]


def discrete_seismic_colors(n=5):
    """From http://colorbrewer2.org/#type=diverging&scheme=RdBu&n=7"""
    if n == 5:
        return list(
            np.array(
                [
                    (5, 113, 176, 256),
                    (146, 197, 222, 256),
                    # (247, 247, 247, 245),  # To make red-white-blue
                    (247, 247, 191, 245),  # To make red-yellow-blue
                    (244, 165, 130, 256),
                    (202, 0, 32, 256),
                ]
            )
            / 256
        )
    elif n == 7:
        # Really this is red-yellow-blue
        # http://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=7
        return list(
            np.array(
                [
                    (69, 117, 199, 256),
                    (145, 191, 219, 256),
                    (224, 243, 248, 256),
                    (255, 255, 191, 255),
                    (254, 224, 144, 256),
                    (252, 141, 89, 256),
                    (215, 48, 39, 256),
                ]
            )
            / 256
        )
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
        (0, 0, 0.3, 1),
        (0, 0, 0.7, 1),
        (0.1, 0.1, 0.9, 1),
        (0.3, 0.3, 0.95, 1),
        (0.6, 0.6, 1, 1),
        (0.85, 0.85, 1, 1),
        (0.92, 0.92, 1, 0.99),
        (0.98, 0.98, 1, 0.98),
        (1, 1, 1, 0.95),
        (1, 0.98, 0.98, 0.98),
        (1, 0.92, 0.92, 0.99),
        (1, 0.85, 0.85, 1),
        (1, 0.6, 0.6, 1),
        (1, 0.3, 0.3, 1),
        (0.9, 0.1, 0.1, 1),
        (0.7, 0, 0, 1),
        (0.3, 0, 0, 1),
    ],
    N=250,
)
plt.register_cmap("seismic_wider", SEISMIC_WIDE2)

seismic_widy_y_list = [
    (0, 0, 0.7, 1),
    (0, 0, 1, 1),
    (0.1, 0.3, 1, 1),
    (0.2, 0.6, 0.95, 1),
    (0.5, 0.8, 0.85, 1),
    (0.8, 0.95, 0.80, 1),
    (0.9, 0.92, 0.78, 1),
    (0.95, 0.95, 0.75, 1),
    (0.95, 0.92, 0.70, 1),
    (0.95, 0.85, 0.65, 1),
    (0.97, 0.65, 0.3, 1),
    (1, 0.3, 0.2, 1),
    (1, 0.1, 0.05, 1),
    (1, 0, 0, 1),
    (0.7, 0, 0, 1),
]
SEISMIC_WIDE_Y = LinearSegmentedColormap.from_list(
    "seismic_wide_y",
    seismic_widy_y_list,
    N=256,
)
plt.register_cmap("seismic_wide_y", SEISMIC_WIDE_Y)
# Also add the reverse colormap with _r in name
SEISMIC_WIDE_Y_R = LinearSegmentedColormap.from_list(
    "seismic_wide_y_r",
    seismic_widy_y_list[::-1],
    N=256,
)
plt.register_cmap("seismic_wide_y_r", SEISMIC_WIDE_Y_R)

DISCRETE_SEISMIC5 = LinearSegmentedColormap.from_list(
    "discrete_seismic5", discrete_seismic_colors(5), N=5
)
plt.register_cmap(cmap=DISCRETE_SEISMIC5)
DISCRETE_SEISMIC7 = LinearSegmentedColormap.from_list(
    "discrete_seismic7", discrete_seismic_colors(7), N=7
)
plt.register_cmap(cmap=DISCRETE_SEISMIC7)

SEISMIC_Y = LinearSegmentedColormap.from_list(
    "seismic_y", discrete_seismic_colors(7), N=250
)
plt.register_cmap(cmap=SEISMIC_Y)
SEISMIC_Y2 = LinearSegmentedColormap.from_list(
    "seismic_y2", discrete_seismic_colors(5), N=250
)
plt.register_cmap(cmap=SEISMIC_Y2)
SEISMIC_WIDE = LinearSegmentedColormap.from_list(
    "seismic_wide",
    [
        (0, 0, 0.3, 1),
        (0, 0, 1, 1),
        (0.6, 0.6, 1, 1),
        (0.9, 0.9, 1, 1),
        (1, 1, 1, 1),
        (1, 0.9, 0.9, 1),
        (1, 0.6, 0.6, 1),
        (1, 0, 0, 1),
        (0.5, 0, 0, 1),
    ],  # Extra white in middle from seismic
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

    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    # N = num_levels
    N = 256
    reg_index = np.linspace(start, stop, N + 1)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, N // 2, endpoint=False),
            np.linspace(midpoint, 1.0, N // 2 + 1, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = LinearSegmentedColormap("shiftedcmap", cdict, N=num_levels or N)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def make_shifted_cmap(
    img=None, vmax=None, vmin=None, cmap_name="seismic", num_levels=None
):
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
        rows.append(ss + ";" + ",".join(map(str, rgb.astype(int))))
    return ":".join(rows)


def make_qgis_cmap(cmap_rgba_arr, outfile, cmap_name):
    # EXAMPLE:
    # make_qgis_cmap(make_dismph_colors().T[::3], "dismph_colors.xml", "dismph")
    c1 = ",".join(map(str, cmap_rgba_arr[0].astype(int)))
    c2 = ",".join(map(str, cmap_rgba_arr[-1].astype(int)))
    stops = cmap_to_qgis(cmap_rgba_arr)
    template = """
<!DOCTYPE qgis_style>
<qgis_style version="1">
  <symbols/>
  <colorramps>
    <colorramp type="gradient" name="{name}" favorite="1">
      <prop k="color1" v="{c1}"/>
      <prop k="color2" v="{c2}"/>
      <prop k="discrete" v="0"/>
      <prop k="rampType" v="gradient"/>
      <prop k="stops" v="{stops}"/>
    </colorramp>
  </colorramps>
</qgis_style>
"""
    with open(outfile, "w") as f:
        f.write(template.format(
            name=cmap_name,
            c1=c1,
            c2=c2,
            stops=stops
        ))


def make_dismph_colors():
    red, green, blue = [], [], []
    for i in range(120):
        red.append(i * 2.13 * 155.0 / 255.0 + 100)
        green.append((119.0 - i) * 2.13 * 155.0 / 255.0 + 100.0)
        blue.append(255)
    for i in range(120):
        red.append(255)
        green.append(i * 2.13 * 155.0 / 255.0 + 100.0)
        blue.append((119 - i) * 2.13 * 155.0 / 255.0 + 100.0)
    for i in range(120):
        red.append((119 - i) * 2.13 * 155.0 / 255.0 + 100.0)
        green.append(255)
        blue.append(i * 2.13 * 155.0 / 255.0 + 100.0)
    return np.vstack((red, green, blue))


DISMPH = LinearSegmentedColormap.from_list("dismph", make_dismph_colors().T / 256)
plt.register_cmap(cmap=DISMPH)


def test_rgbmat(plot=True):
    """make a square showing the dismph color gradient"""
    rgbmat = make_dismph_colors()
    N = rgbmat.shape[1]
    gradient = np.ones((1, N)) * (np.ones(N).reshape((N, 1)) / N)
    square_grad = gradient[:, :, np.newaxis] * (rgbmat.T)[:, np.newaxis, :]
    if plot:
        plt.figure()
        plt.imshow(square_grad)
        plt.colorbar()
    return square_grad
