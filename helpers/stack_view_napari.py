#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
import hdf5plugin
import h5py

import napari

import sys

fname, dset = sys.argv[1:3]

# create image
x = np.linspace(0, 5, 256)
y = np.linspace(0, 5, 256)[:, np.newaxis]
img = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
img_stack = np.stack([x * img for x in range(1, 5)])
hf = h5py.File(fname)
ds = hf[dset]


# add it to the viewer
# viewer = napari.view_image(img, colormap="viridis")
# viewer = napari.view_image(img_stack, colormap="viridis")
viewer = napari.view_image(ds, colormap="viridis")
layer = viewer.layers[-1]

# create mpl figure with subplots
mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)
# (line,) = ax.plot(layer.data[1, 123, :])  # linescan through the middle of the image
(line,) = ax.plot(layer.data[:, 123, 123])  # linescan through the middle of the image

# add the figure to the viewer as a FigureCanvas widget
viewer.window.add_dock_widget(FigureCanvas(mpl_fig))


# connect a callback that updates the line plot when
# the user clicks on the image
@layer.mouse_drag_callbacks.append
def profile_lines_drag(layer, event):
    try:
        print(layer.data.shape)
        # (5, 256, 256)
        print(event.position)
        # (0.0, 53.49606383747026, 37.19011420545798)
        lay, row, col = map(int, event.position)

        dd = layer.data[:, row, col]
        # dd = layer.data[lay, row, :]
        line.set_ydata(dd)
        line.figure.canvas.draw()
    except IndexError:
        pass


napari.run()
