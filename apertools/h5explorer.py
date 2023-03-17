from io import BytesIO

import h5py
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt


class HDF5Explorer:
    """Class which maps an HDF5 file and allows tab-completion to explore datasets."""
    def __init__(self, hdf5_filepath: str, load_less_than: float = 1e3):
        self.hdf5_filepath = hdf5_filepath
        self._hf = h5py.File(hdf5_filepath, "r")
        self._root_group = HDF5GroupExplorer(
            self._hf["/"], load_less_than=load_less_than
        )

    def close(self):
        self._hf.close()

    def __getattr__(self, name):
        return getattr(self._root_group, name)

    def __dir__(self):
        return self._root_group.__dir__()

    def __repr__(self):
        return f"HDF5Explorer({self.hdf5_filepath})"


class HDF5GroupExplorer:
    def __init__(self, group: h5py.Group, load_less_than: float = 1e3):
        self._group = group
        self._attr_cache = {}
        self._populate_attr_cache(load_less_than)

    @property
    def group_path(self) -> str:
        return self._group.name

    def _populate_attr_cache(self, load_less_than: float = 1e3):
        for name, item in self._group.items():
            if isinstance(item, h5py.Group):
                self._attr_cache[name] = HDF5GroupExplorer(item)
            elif isinstance(item, h5py.Dataset):
                if item.size < load_less_than:
                    self._attr_cache[name] = item[()]
                else:
                    self._attr_cache[name] = item
            else:
                self._attr_cache[name] = item

    def __getattr__(self, name):
        if name not in self._attr_cache:
            raise AttributeError(
                f"'{name}' not found in the group '{self.group_path}'."
            )
        return self._attr_cache[name]

    def __dir__(self):
        return list(self._attr_cache.keys())


def create_explorer_widget(hf: h5py.File, load_less_than: float = 1e3):
    """Make a widget in Jupyter to explore a h5py file.

    Example
    -------
    >>> hf = h5py.File("file.h5", "r")
    >>> create_explorer_widget(hf)
    """
    def _make_thumbnail(image):
        # Create a thumbnail of the dataset
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image, cmap="gray", vmax=np.nanpercentile(image, 99))
        ax.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        # Display the thumbnail in an Image widget
        return widgets.Image(value=buf.read(), format='png')

    def _add_widgets(item: Any, level: int = 0):
        """Recursively add widgets to the accordion widget."""
        if isinstance(item, h5py.Group):
            # Add a new accordion widget for the group
            accordion = widgets.Accordion(selected_index=None)
            for key, value in item.items():
                widget = _add_widgets(value, level + 1)
                accordion.children += (widget,)
                accordion.set_title(len(accordion.children) - 1, key)
            return accordion

        # Once we're at a leaf node, add a widget for the dataset
        elif isinstance(item, h5py.Dataset):
            attributes = [f"<b>{k}:</b> {v}" for k, v in item.attrs.items()]
            content = f"Type: {item.dtype}<br>Shape: {item.shape}<br>"
            content += "<br>".join(attributes)
            if item.size < load_less_than:
                content += f"<br>Value: {item[()]}"
            html_widget = widgets.HTML(content)

            if not item.ndim == 2 or not item.dtype == np.complex64:
                return html_widget
            # If the dataset is a 2D complex array, make a thumbnail
            image_widget = _make_thumbnail(np.abs(item[::5, ::10]))
            return widgets.VBox([image_widget, html_widget])

        else:
            # Other types of items
            return widgets.HTML(f"{item}")

    # Now add everything starting at the root
    return _add_widgets(hf, 0)
