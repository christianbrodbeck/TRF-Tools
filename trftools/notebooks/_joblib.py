"""Memory backend for joblib

Backend that will keep objects in memory

This backend is registered with :mod:joblib: when importing
``trftools.notebooks``. Use it like::

    from trftools import notebooks  # register backend
    import joblib
    memory = joblib.Memory('.', 'memory')


Location has to be given, otherwise :mod:`joblib` will not initialize the
backend.

.. warning::
    Cached results are not copied; when a retrieved item is modified, this
    modification will persist the next time the cached function is called
"""
from collections import namedtuple
import os
from pathlib import Path

import joblib
from joblib.memory import StoreBackendBase


CacheItemInfo = namedtuple('CacheItemInfo', 'path size last_access')


def walk(path: Path, base: dict):
    out = []
    for key, value in base.items():
        if isinstance(value, dict):
            out.extend(walk(path / key, value))
        else:
            out.append(CacheItemInfo(str(path / key), 1, 1))
    return out


def to_path(path):
    if isinstance(path, list):
        path = Path(os.path.join(*path))
    elif isinstance(path, str):
        path = Path(path)
    elif not isinstance(path, Path):
        raise TypeError(path)
    return path


class MemoryBackend(StoreBackendBase):
    __store = None

    def configure(self, location, verbose=0, backend_options=None):
        self.__store = {}

    def get_dir(self, path: Path):
        path = to_path(path)
        curdir = self.__store
        for key in path.parts:
            if key not in curdir:
                curdir[key] = {}
            curdir = curdir[key]
        return curdir

    def create_location(self, location):
        self.get_dir(Path(location))

    def clear(self):
        self.__store.clear()

    def clear_location(self, location):
        path = Path(location)
        parent = self.get_dir(path.parent)
        del parent[path.name]

    def get_items(self):
        return walk(Path(self.location), self.__store)

    def _open_item(self, f, mode):
        print("OPEN", f, mode)

    def _item_exists(self, location):
        print("ITEM_EXITS", location)

    def _move_item(self, src, dst):
        print("MOVE", src, dst)

    def store_cached_func_code(self, path, func_code=None):
        if func_code is not None:
            self.get_dir(path)['func_code'] = func_code

    def get_cached_func_code(self, path):
        try:
            return self.get_dir(path)['func_code']
        except KeyError:
            raise IOError

    def dump_item(self, path, item, verbose=1):
        self.get_dir(path)['output'] = item

    def load_item(self, path, verbose=1, msg=None):
        return self.get_dir(path)['output']

    def store_metadata(self, path, metadata):
        self.get_dir(path)['metadata'] = metadata

    def get_metadata(self, path):
        return self.get_dir(path)['metadata']

    def contains_item(self, path):
        path = to_path(path)
        curdir = self.__store
        for key in path.parts:
            if key not in curdir:
                return False
            curdir = curdir[key]
        return True


joblib.register_store_backend('memory', MemoryBackend)
