import os
import numpy as np
from warnings import warn


class Storage:

    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.files = os.listdir(path)

    def generate_name(self, name=None):
        """generates a unique name for storing an array"""
        if name is None:
            name = "array"
        existing_files = set(self.files)
        i = 0
        while True:
            name_loc = f"{name}{f"_{i}" if i > 0 else ''}.npy"
            if name_loc not in existing_files:
                return name_loc
            i += 1

    def store_array(self, array, name=None, ignore_existing=False):
        name = self.generate_name(name)
        if os.name in self.files and not ignore_existing:
            warn(f"File {name} already exists in storage {self.path}. Overwriting.")
        path = os.path.join(self.path, name)
        np.save(path, array)
        self.files.append(name)
        return StoredArray(name=name, storage=self)


    def get_array(self, name):
        if name not in self.files:
            raise FileNotFoundError(f"File {name} not found in storage {self.path}.")
        return np.load(os.path.join(self.path, name))


class DummyStorage(Storage):

    def __init__(self, path=None):
        self.arrays = {}
        self.files = []

    def store_array(self, array, name=None, ignore_existing=False):
        if name is None:
            name = f"array_{len(self.arrays)}"
        if name in self.arrays and not ignore_existing:
            warn(f"Array {name} already exists in storage. Overwriting.")
        self.arrays[name] = array
        self.files.append(name)
        return StoredArray(name=name, storage=self)

    def get_array(self, name):
        if name not in self.arrays:
            raise KeyError(f"Array {name} not found in storage.")
        return self.arrays[name]


class StoredArray:

    def __init__(self, name, storage):
        self.name = name
        self.storage = storage

    def get(self):
        return self.storage.get_array(self.name)
