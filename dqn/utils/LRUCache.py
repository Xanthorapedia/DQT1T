import collections
import os
from PIL import Image
import numpy as np

# https://www.kunxi.org/blog/2014/05/lru-cache-in-python/


class LRUCache:
    def __init__(self, capacity, save_dir=None):
        self.cap = capacity
        self.cache = collections.OrderedDict()
        self.save_dir = save_dir

    def get(self, key):
        try:
            val = self.cache.pop(key)
        except KeyError:
            # try to load from disk
            file_dir = '{}/{}.png'.format(self.save_dir, key)
            try:
                img = Image.open(file_dir)
                val = np.array(img)[:, :, 0:3]
            except FileNotFoundError:
                val = None
        self.cache[key] = val
        return val

    def set(self, key, val, replace_on_disk=False):
        try:
            self.cache.pop(key)
        except KeyError:
            # clean out space if necessary
            if len(self.cache) >= self.cap:
                old_k, old_v = self.cache.popitem(last=False)
                # save to disk
                self._save(old_k, old_v, replace_on_disk)
        self.cache[key] = val
        return val

    def save(self, replace=False):
        for key, val in self.cache.items():
            self._save(key, val, replace)

    def _save(self, key, val, replace_on_disk=False, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        # create dir if DNE
        if self.save_dir is not None:
            if not os.path.isdir(self.save_dir):
                os.mkdir(self.save_dir)
        file_dir = '{}/{}.png'.format(save_dir, key)
        # save if no such file or forced replace
        if (not os.path.isfile(file_dir) or replace_on_disk) and val is not None:
            img = Image.fromarray(val)
            img.save(file_dir)
