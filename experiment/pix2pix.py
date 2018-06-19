#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: pix2pix.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import argparse
import platform
import scipy.misc
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from lib.model.pix2pix import Pix2Pix
from lib.dataflow.facades import ImagePair


if platform.node() == 'Qians-MacBook-Pro.local':
    DATA_PATH = '/Users/gq/workspace/Dataset/CMP_facade_DB_base/'
    SAVE_PATH = '/Users/gq/tmp/dlout/pix2pix/'
    # RESULT_PATH = '/Users/gq/tmp/ram/center/result/'
elif platform.node() == 'arostitan':
    DATA_PATH = '/home/qge2/workspace/data/facades/'
    SAVE_PATH = '/home/qge2/workspace/data/out/pix2pix/'
else:
    # DATA_PATH = 'E://Dataset//MNIST//'
    SAVE_PATH = 'E:/tmp/pix2pix/'
    # RESULT_PATH = 'E:/tmp/ram/trans/result/'

def test_im_normalize(im):
    return scipy.misc.imresize(im, [256, 256]) * 2.0 / 255.0 - 1.0

def im_normalize(im):
    return scipy.misc.imresize(im, [286, 286]) * 2.0 / 255.0 - 1.0

def im_preprocess(im_pair):
    # im_pair = [scipy.misc.imresize(im, [286, 286]) * 2.0 / 255.0 - 1.0
    #            for im in im_pair]
    h = int(np.ceil(np.random.uniform(1e-2, 286 - 256)))
    w = int(np.ceil(np.random.uniform(1e-2, 286 - 256)))
    im_pair = [im[:, h: h + 256, w: w + 256, :] for im in im_pair]

    if np.random.random() > 0.5:
        im_pair = [im[:,:,::-1,:] for im in im_pair]

    return im_pair


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', action='store_true',
                        help='Test')
    
    return parser.parse_args()

if __name__ == '__main__':
    FLAGS = get_args()
    # im = scipy.misc.imread(os.path.join(DATA_PATH, 'cmp_b0001.png'))
    train_path = os.path.join(DATA_PATH, 'base')
    train_data = ImagePair(
        data_dir_pair=[train_path, train_path],
        ext_name_pair=['.png', '.jpg'],
        batch_dict_name=['input', 'label'],
        pf=im_normalize,
        pair_pf=im_preprocess)
    train_data.setup(epoch_val=0, batch_size=1)

    test_path = os.path.join(DATA_PATH, 'extend')
    test_data = ImagePair(
        data_dir_pair=[test_path, test_path],
        ext_name_pair=['.png', '.jpg'],
        batch_dict_name=['input', 'label'],
        pf=test_im_normalize,
        shuffle=False,
        # pair_pf=im_preprocess
        )
    test_data.setup(epoch_val=0, batch_size=10)

    model = Pix2Pix(x_n_channel=3, y_n_channel=3)
    model.create_model()
    train_g_op = model.train_op('G')
    train_d_op = model.train_op('D')
    g_loss_op = model.get_loss('G')
    d_loss_op = model.get_loss('D')
    train_summary_op = model.get_summary('train')
    test_summary_op = model.get_summary('test')
    sample_op = model.layers['fake']

    writer = tf.summary.FileWriter(SAVE_PATH)
    saver = tf.train.Saver()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())

        if FLAGS.train:
            writer.add_graph(sess.graph)

            for i in range(0, 80000):
                batch_data = train_data.next_batch_dict()
                sess.run(train_d_op,
                         feed_dict={model.lr: 0.0002,
                                    model.keep_prob: 0.5,
                                    model.image: batch_data['input'],
                                    model.label: batch_data['label']})

                sess.run(train_g_op, 
                         feed_dict={model.lr: 0.0002,
                                    model.keep_prob: 0.5,
                                    model.image: batch_data['input'],
                                    model.label: batch_data['label']})

                sess.run(train_g_op, 
                         feed_dict={model.lr: 0.0002,
                                    model.keep_prob: 0.5,
                                    model.image: batch_data['input'],
                                    model.label: batch_data['label']})

                g_loss, d_loss, cur_summary = sess.run(
                    [g_loss_op, d_loss_op, train_summary_op],
                    feed_dict={
                               model.keep_prob: 0.5,
                               model.image: batch_data['input'],
                               model.label: batch_data['label']})

                if i % 200 == 0:
                    saver.save(sess, '{}step_{}'.format(SAVE_PATH, i), global_step=i)
                    writer.add_summary(cur_summary, i)

                    batch_data = test_data.next_batch_dict()
                    test_summary = sess.run(
                        test_summary_op,
                        feed_dict={
                                   model.keep_prob: 0.5,
                                   model.image: batch_data['input'],
                                   model.label: batch_data['label']})
                    writer.add_summary(test_summary, i)
                print('step: {}, g_loss: {}, d_loss: {}'.format(i, g_loss, d_loss))

        if FLAGS.test:
            saver.restore(sess, '{}step_{}-{}'.format(SAVE_PATH, 34200, 34200))

            batch_data = test_data.next_batch_dict()
            sample = sess.run(sample_op,
                              feed_dict={
                                         model.keep_prob: 0.5,
                                         model.image: batch_data['input'],
                                        })
            # print(np.array(np.squeeze(sample[0])).shape)
            for idx, c_sample in enumerate(sample):
                scipy.misc.imsave('{}/im_{}.png'.format(SAVE_PATH, idx), np.squeeze(c_sample))


