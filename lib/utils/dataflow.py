#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dataflow.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
from datetime import datetime
import numpy as np
import scipy.misc

import lib.utils.utils as utils

def identity(inputs):
    return inputs

def fill_pf_list(pf_list, n_pf, fill_with_fnc=identity):
    if pf_list == None:
        return [identity for i in range(n_pf)]
    new_list = []
    for pf in pf_list:
        if not pf:
            pf = identity
        new_list.append(pf)
    pf_list = new_list

    pf_list = utils.make_list(pf_list)
    if len(pf_list) > n_pf:
        raise ValueError('Invalid number of preprocessing functions')
    pf_list = pf_list + [fill_with_fnc for i in range(n_pf - len(pf_list))]
    return pf_list

def load_image(im_path, read_channel=None, pf=identity):
    if read_channel is None:
        im = scipy.misc.imread(im_path)
    elif read_channel == 3:
        im = scipy.misc.imread(im_path, mode='RGB')
    else:
        im = scipy.misc.imread(im_path, flatten=True)

    if len(im.shape) < 3:
        im = pf(im)
        im = np.reshape(im, [1, im.shape[0], im.shape[1], 1])
    else:
        im = pf(im)
        im = np.reshape(im, [1, im.shape[0], im.shape[1], im.shape[2]])
    return im

def get_file_list(file_dir, file_ext, sub_name=None):
    re_list = []

    if sub_name is None:
        return np.array([os.path.join(root, name)
            for root, dirs, files in os.walk(file_dir) 
            for name in sorted(files) if name.endswith(file_ext)])
    else:
        return np.array([os.path.join(root, name)
            for root, dirs, files in os.walk(file_dir) 
            for name in sorted(files) if name.endswith(file_ext) and sub_name in name])

_RNG_SEED = None

def get_rng(obj=None):
    """
    This function is copied from `tensorpack
    <https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/utils.py>`__.
    Get a good RNG seeded with time, pid and the object.
    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return np.random.RandomState(seed)