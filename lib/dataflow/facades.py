#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: facades.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
from scipy import misc
from tensorcv.dataflow.base import RNGDataFlow
from tensorcv.dataflow.common import get_file_list

def identity(inputs):
    return inputs

def load_image(im_path, read_channel=None, pf=identity):
    if read_channel is None:
        im = misc.imread(im_path)
    elif read_channel == 3:
        im = misc.imread(im_path, mode='RGB')
    else:
        im = misc.imread(im_path, flatten=True)

    if len(im.shape) < 3:
        im = pf(im)
        im = np.reshape(im, [1, im.shape[0], im.shape[1], 1])
    else:
        im = pf(im)
        im = np.reshape(im, [1, im.shape[0], im.shape[1], im.shape[2]])
    return im

class ImagePair(RNGDataFlow):
    def __init__(self,
                 data_dir_pair,
                 ext_name_pair,
                 batch_dict_name,
                 pf=identity,
                 pair_pf=identity,
                 shuffle=True):
        self._batch_dict_name = batch_dict_name
        self._pf = pf
        self._pair_pf = pair_pf
        self._shuffle = shuffle

        assert isinstance(ext_name_pair, list)
        assert len(ext_name_pair) == 2

        assert isinstance(data_dir_pair, list)
        assert len(data_dir_pair) == 2

        assert isinstance(batch_dict_name, list)
        assert len(batch_dict_name) == 2

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
        for k in range(start, end):
            im_list = []
            for i in range(0, 2):
                im_path = self._im_list[i][k]
                # print(im_path)
                im = load_image(im_path, read_channel=None, pf=self._pf)
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

