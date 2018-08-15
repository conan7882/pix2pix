#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: trainer.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf


def display(global_step,
            step,
            scaler_sum_list,
            name_list,
            collection,
            summary_val=None,
            summary_writer=None,
            ):
    print('[step: {}]'.format(global_step), end='')
    for val, name in zip(scaler_sum_list, name_list):
        print(' {}: {:.4f}'.format(name, val * 1. / step), end='')
    print('')
    if summary_writer is not None:
        s = tf.Summary()
        for val, name in zip(scaler_sum_list, name_list):
            s.value.add(tag='{}/{}'.format(collection, name),
                        simple_value=val * 1. / step)
        summary_writer.add_summary(s, global_step)
        if summary_val is not None:
            summary_writer.add_summary(summary_val, global_step)

class Trainer(object):
    def __init__(self, train_model, train_data, init_lr=2e-4):
        self._train_data = train_data
        self._t_model = train_model
        self._train_g_op = train_model.train_op('G')
        self._train_d_op = train_model.train_op('D')
        self._d_loss_op = train_model.get_loss('D')
        # self._g_loss_op = train_model.get_loss('G')
        self._g_loss_op = train_model.fake_loss
        self._l1_loss_op = train_model.l1_loss
        self._train_summary_op = train_model.get_train_summary()

        self.global_step = 0
        self.epoch_id = 0
        self.init_lr = init_lr

    def train_epoch(self, sess, keep_prob=1.0, summary_writer=None):
        display_name_list = ['d_loss', 'g_loss', 'l1_loss']

        cur_epoch = self._train_data.epochs_completed

        step = 0
        d_loss_sum = 0
        g_loss_sum = 0
        l1_loss_sum = 0
        self.epoch_id += 1
        while cur_epoch == self._train_data.epochs_completed:
            self.global_step += 1
            step += 1

            batch_data = self._train_data.next_batch_dict()
            im = batch_data['input']
            label = batch_data['label']

            sess.run(self._train_d_op,
                     feed_dict={self._t_model.lr: self.init_lr,
                                self._t_model.keep_prob: keep_prob,
                                self._t_model.image: im,
                                self._t_model.label: label})

            sess.run(self._train_g_op, 
                     feed_dict={self._t_model.lr: self.init_lr,
                                self._t_model.keep_prob: keep_prob,
                                self._t_model.image: im,
                                self._t_model.label: label})

            g_loss, d_loss, l1_loss = sess.run(
                [self._g_loss_op, self._d_loss_op, self._l1_loss_op],
                feed_dict={self._t_model.keep_prob: keep_prob,
                           self._t_model.image: im,
                           self._t_model.label: label})

            d_loss_sum += d_loss
            g_loss_sum += g_loss
            l1_loss_sum += l1_loss

            if step % 100 == 0:
                cur_summary = sess.run(
                    self._train_summary_op, 
                    feed_dict={self._t_model.keep_prob: keep_prob,
                               self._t_model.image: im,
                               self._t_model.label: label})

                display(self.global_step,
                    step,
                    [d_loss_sum, g_loss_sum, l1_loss_sum],
                    display_name_list,
                    'train',
                    summary_val=cur_summary,
                    summary_writer=summary_writer)

        print('==== epoch: {}, lr:{} ===='.format(cur_epoch, self.init_lr))
        cur_summary = sess.run(
            self._train_summary_op, 
            feed_dict={self._t_model.keep_prob: keep_prob,
                       self._t_model.image: im,
                       self._t_model.label: label})

        display(self.global_step,
                step,
                [d_loss_sum, g_loss_sum, l1_loss_sum],
                display_name_list,
                'train',
                summary_val=cur_summary,
                summary_writer=summary_writer)
