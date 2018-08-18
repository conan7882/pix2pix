#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: pix2pix.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import argparse
import scipy.misc
import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

from lib.model.pix2pix import Pix2Pix
import loader as loader
from lib.helper.trainer import Trainer
from lib.helper.generator import Generator


SAVE_PATH = '/home/qge2/workspace/data/out/pix2pix/new/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', action='store_true',
                        help='Test')

    parser.add_argument('--load', type=int, default=199,
                        help='load pre-trained model')
    parser.add_argument('--dataset', type=str, default='facades',
                        help='type of dataset')
    
    return parser.parse_args()

def load_data():
    FLAGS = get_args()
    if FLAGS.dataset == 'maps':
        train_data, test_data = loader.load_maps(batch_size=1)
    elif FLAGS.dataset == 'mapsreverse':
        train_data, test_data = loader.load_maps(batch_size=1, reverse=True)
    elif FLAGS.dataset == 'shoes':
        train_data, test_data = loader.load_shoes(batch_size=1)
    elif FLAGS.dataset == 'facades':
        # FLAGS.dataset = 'facades'
        train_data, test_data = loader.load_facades(batch_size=1)
    else:
        raise ValueError('No dataset {}!'.format(FLAGS.dataset))

    return train_data, test_data, FLAGS.dataset

def train():
    FLAGS = get_args()

    train_data, test_data, data_name = load_data()
    save_path = os.path.join(SAVE_PATH, data_name)
    save_path += '/'

    train_model = Pix2Pix(x_n_channel=3, y_n_channel=3)
    train_model.create_train_model()

    test_model = Pix2Pix(x_n_channel=3, y_n_channel=3)
    test_model.create_test_model()

    trainer = Trainer(train_model, train_data, init_lr=2e-4)
    generator = Generator(test_model)
    writer = tf.summary.FileWriter(save_path)
    saver = tf.train.Saver()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_id in range(200):
            trainer.train_epoch(sess, keep_prob=0.5, summary_writer=writer)
            generator.generate_step(sess, test_data, summary_writer=writer)
            saver.save(sess, '{}epoch_{}'.format(save_path, epoch_id))
        saver.save(sess, '{}epoch_{}'.format(save_path, epoch_id))

def test():
    FLAGS = get_args()
    train_data, test_data, data_name = load_data()
    save_path = os.path.join(SAVE_PATH, data_name)
    save_path += '/'

    test_model = Pix2Pix(x_n_channel=3, y_n_channel=3)
    test_model.create_test_model()

    generator = Generator(test_model)
    saver = tf.train.Saver()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '{}epoch_{}'.format(save_path, FLAGS.load))
        generator.generate_epoch(sess, test_data, save_path=save_path)


if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.train:
        train()
    if FLAGS.test:
        test()
    