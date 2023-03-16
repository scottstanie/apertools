import h5py


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

