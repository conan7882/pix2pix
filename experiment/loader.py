#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: loader.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import scipy.misc
import numpy as np

import sys
sys.path.append('../')

from lib.dataflow.images import ImagePair


def im_preprocess(im_pair):
    h = int(np.ceil(np.random.uniform(1e-2, 286 - 256)))
    w = int(np.ceil(np.random.uniform(1e-2, 286 - 256)))
    # im_pair = [im[:, h: h + 256, w: w + 256, :] for im in im_pair]
    im_pair = [im[h: h + 256, w: w + 256, :] for im in im_pair]

    # flip
    if np.random.random() > 0.5:
        # im_pair = [im[:,:,::-1,:] for im in im_pair]
        im_pair = [im[:,::-1,:] for im in im_pair]

    return im_pair

def load_facades(batch_size=1):
    def train_pf(im):
        resize_norm_im = scipy.misc.imresize(im, [286, 286]) * 2.0 / 255.0 - 1.0
        return np.clip(resize_norm_im, -1., 1.)

    data_path = '/home/qge2/workspace/data/facades/'
    # data_path = 'E:\Dataset/facade/CMP_facade_DB_extended/'

    train_path = os.path.join(data_path, 'base')
    train_data = ImagePair(
        data_dir_pair=[train_path, train_path],
        ext_name_pair=['.png', '.jpg'],
        batch_dict_name=['input', 'label'],
        pf=[train_pf, train_pf],
        pair_pf=im_preprocess)
    train_data.setup(epoch_val=0, batch_size=batch_size)

    def test_pf(im):
        resize_norm_im = scipy.misc.imresize(im, [256, 256]) * 2.0 / 255.0 - 1.0
        return np.clip(resize_norm_im, -1., 1.)

    test_path = os.path.join(data_path, 'extended')
    test_data = ImagePair(
        data_dir_pair=[test_path, test_path],
        ext_name_pair=['.png', '.jpg'],
        batch_dict_name=['input', 'label'],
        pf=[test_pf, test_pf],
        shuffle=False,
        # pair_pf=im_preprocess
        )
    test_data.setup(epoch_val=0, batch_size=1)

    return train_data, test_data

def load_maps(batch_size=1, reverse=False):
    # data_path = 'E:/Dataset/maps/maps/'
    data_path = '/home/qge2/workspace/data/maps/'
    # data_path = 'E:\Dataset/facade/CMP_facade_DB_extended/'

    if reverse == True:
        batch_dict_name = ['label', 'input']
    else:
        batch_dict_name = ['input', 'label']

    def train_input_pf(im):
        im = im[:, 600:, :]
        resize_norm_im = scipy.misc.imresize(im, [286, 286]) * 2.0 / 255.0 - 1.0
        return np.clip(resize_norm_im, -1., 1.)

    def train_label_pf(im):
        im = im[:, :600, :]
        resize_norm_im = scipy.misc.imresize(im, [286, 286]) * 2.0 / 255.0 - 1.0
        return np.clip(resize_norm_im, -1., 1.)

    train_path = os.path.join(data_path, 'train')
    train_data = ImagePair(
        data_dir_pair=[train_path, train_path],
        ext_name_pair=['.jpg', '.jpg'],
        batch_dict_name=batch_dict_name,
        pf=[train_input_pf, train_label_pf],
        pair_pf=im_preprocess)
    train_data.setup(epoch_val=0, batch_size=batch_size)

    def test_input_pf(im):
        im = im[:, 600:, :]
        resize_norm_im = scipy.misc.imresize(im, [256, 256]) * 2.0 / 255.0 - 1.0
        return np.clip(resize_norm_im, -1., 1.)

    def test_label_pf(im):
        im = im[:, :600, :]
        resize_norm_im = scipy.misc.imresize(im, [256, 256]) * 2.0 / 255.0 - 1.0
        return np.clip(resize_norm_im, -1., 1.)

    test_path = os.path.join(data_path, 'val')
    test_data = ImagePair(
        data_dir_pair=[test_path, test_path],
        ext_name_pair=['.jpg', '.jpg'],
        batch_dict_name=batch_dict_name,
        pf=[test_input_pf, test_label_pf],
        shuffle=False,
        # pair_pf=im_preprocess
        )
    test_data.setup(epoch_val=0, batch_size=1)

    return train_data, test_data

def load_shoes(batch_size=1):
    # data_path = 'E:/Dataset/maps/maps/'
    data_path = '/home/qge2/workspace/data/edges2shoes/'
    # data_path = 'E:/Dataset/edges2shoes/edges2shoes/'

    def input_pf(im):
        im = im[:, :256:, :]
        resize_norm_im = scipy.misc.imresize(im, [256, 256]) * 2.0 / 255.0 - 1.0
        return np.clip(resize_norm_im, -1., 1.)

    def label_pf(im):
        im = im[:, 256:, :]
        resize_norm_im = scipy.misc.imresize(im, [256, 256]) * 2.0 / 255.0 - 1.0
        return np.clip(resize_norm_im, -1., 1.)

    train_path = os.path.join(data_path, 'train')
    train_data = ImagePair(
        data_dir_pair=[train_path, train_path],
        ext_name_pair=['.jpg', '.jpg'],
        batch_dict_name=['input', 'label'],
        pf=[input_pf, label_pf],
        # pair_pf=im_preprocess
        )
    train_data.setup(epoch_val=0, batch_size=batch_size)

    test_path = os.path.join(data_path, 'val')
    test_data = ImagePair(
        data_dir_pair=[test_path, test_path],
        ext_name_pair=['.jpg', '.jpg'],
        batch_dict_name=['input', 'label'],
        pf=[input_pf, label_pf],
        shuffle=False,
        # pair_pf=im_preprocess
        )
    test_data.setup(epoch_val=0, batch_size=1)

    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = load_shoes(batch_size=1)
    import matplotlib.pyplot as plt
    batch_data = test_data.next_batch_dict()
    im = batch_data['input'][0]
    label = batch_data['label'][0]
    print(im)
    plt.figure()
    plt.imshow(np.squeeze(im))

    plt.figure()
    plt.imshow(np.squeeze(label))
    plt.show()
