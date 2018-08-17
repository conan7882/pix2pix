#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: generator.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np
import scipy.misc
import tensorflow as tf


class Generator(object):
    def __init__(self, generate_model):

        self._g_model = generate_model
        self._test_summary_op = generate_model.get_test_summary()
        self.global_step = 0

    def generate_step(self, sess, dataflow, summary_writer=None):
        self.global_step += 1
        batch_data = dataflow.next_batch_dict()
        im = batch_data['input']
        label = batch_data['label']

        cur_summary = sess.run(
            self._test_summary_op, 
            feed_dict={
                       self._g_model.image: im,
                       self._g_model.label: label})
        if summary_writer is not None:
            summary_writer.add_summary(cur_summary, self.global_step)

    def generate_epoch(self, sess, test_data, save_path=None):
        test_data.setup(epoch_val=0, batch_size=1)

        im_id = 0
        while test_data.epochs_completed < 1:
            batch_data = test_data.next_batch_dict()
            im = batch_data['input']
            file_names = test_data.batch_file_name[0]

            result_image = sess.run(
                self._g_model.layers['fake'], 
                feed_dict={self._g_model.image: im})
            if save_path is not None:
                for im, im_name in zip(result_image, file_names):
                    drive, path_and_file = os.path.splitdrive(im_name)
                    path, file = os.path.split(path_and_file)
                    im_name, ext = os.path.splitext(file)
                    scipy.misc.imsave('{}im_{}.png'.format(save_path, im_name),
                                      np.squeeze(im))
                    print('Result image {}im_{}.png saved!'.format(save_path, im_name))
                    im_id += 1



            



            
