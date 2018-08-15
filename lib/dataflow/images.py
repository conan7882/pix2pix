#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: images.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np
from scipy import misc
from lib.utils.dataflow import get_rng, get_file_list, load_image
# from tensorcv.dataflow.base import RNGDataFlow
# from tensorcv.dataflow.common import get_file_list

def identity(inputs):
    return inputs

class ImagePair(object):
    def __init__(self,
                 data_dir_pair,
                 ext_name_pair,
                 batch_dict_name,
                 pf=identity,
                 pair_pf=identity,
                 shuffle=True):
        self._batch_dict_name = batch_dict_name
        if not isinstance(pf, list):
            pf = [pf]
        if len(pf) < 2:
            pf.append(identity)
        self._pf = pf
        self._pair_pf = pair_pf
        self._shuffle = shuffle

        assert isinstance(ext_name_pair, list)
        assert len(ext_name_pair) == 2

        assert isinstance(data_dir_pair, list)
        assert len(data_dir_pair) == 2

        assert isinstance(batch_dict_name, list)
        assert len(batch_dict_name) == 2

        self.batch_file_name = [[] for i in range(2)]
        self.setup(epoch_val=0, batch_size=1)
        self._load_file_list(data_dir_pair, ext_name_pair)
        self._data_id = 0

    def size(self):
        return len(self._im_list[0])

    def next_batch(self):
        assert self._batch_size <= self.size(), \
            "batch_size cannot be larger than data size"

        if self._data_id + self._batch_size > self.size():
            start = self._data_id
            end = self.size()
        else:
            start = self._data_id
            self._data_id += self._batch_size
            end = self._data_id
        # batch_file_range = range(start, end)
        batch_data = self._load_data(start, end)

        if end == self.size():
            self._epochs_completed += 1
            self._data_id = 0
            if self._shuffle:
                self._suffle_file_list()
        return batch_data

    def next_batch_dict(self):
        batch_data = self.next_batch()
        batch_dict = {name: data for name, data\
            in zip(self._batch_dict_name, batch_data)}
        return batch_dict

    def _load_data(self, start, end):
        data_list = [[], []]
        self.batch_file_name[0] = []
        self.batch_file_name[1] = []
        for k in range(start, end):
            im_list = []
            for i in range(0, 2):
                im_path = self._im_list[i][k]
                drive, path_and_file = os.path.splitdrive(im_path)
                path, file = os.path.split(path_and_file)
                im_name, ext = os.path.splitext(file)
                self.batch_file_name[i].append(im_name)
                # print(im_path)
                im = load_image(im_path, read_channel=None, pf=self._pf[i])
                im_list.append(im)
            im_list = self._pair_pf(im_list)
            for idx, im in enumerate(im_list):
                data_list[idx].extend(im)
        for i in range(0, 2):   
            data_list[i] = np.array(data_list[i])
        return data_list

    def _load_file_list(self, data_dir_pair, ext_name_pair):
        self._im_list = []
        for idx, (im_dir, im_ext) in enumerate(zip(data_dir_pair, ext_name_pair)):
            cur_list = sorted(get_file_list(im_dir, im_ext))
            self._im_list.append(cur_list)
        self._im_list = np.array(self._im_list)

        if self._shuffle:
            self._suffle_file_list()

    def _suffle_file_list(self):
        idxs = np.arange(self.size())
        self.rng.shuffle(idxs)
        for i in range(0, 2):
            self._im_list[i] = self._im_list[i][idxs]

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def setup(self, epoch_val, batch_size, **kwargs):
        self._epochs_completed  = epoch_val
        self._batch_size = batch_size
        self.rng = get_rng(self)
