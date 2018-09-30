#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: pix2pix.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
import tensorcv.models.layers as layers
from tensorcv.models.base import BaseModel

import lib.model.layers as L

INIT_W_STD = 0.02

class Pix2Pix(BaseModel):
    def __init__(self, x_n_channel, y_n_channel):
        self._x_n_c = x_n_channel
        self._y_n_c = y_n_channel

    def _create_input(self):
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(
            tf.float32,
            name='image',
            shape=[None, None, None, self._x_n_c])
        self.label = tf.placeholder(
            tf.float32,
            name='label',
            shape=[None, None, None, self._y_n_c])

    def create_model(self):
        self.layers = {}
        self._create_input()

        self.layers['fake'] = self._generator(self.image)
        self.layers['d_real'] = self._discriminator(self.image, self.label)
        self.layers['d_fake'] = self._discriminator(self.image, self.layers['fake'])

        self.setup_summary()
        # self.layers['d_fake_freeze_g'] = self._discriminator(
        #     self.image, tf.stop_gradient(self.layers['fake']))

    def train_op(self, name):
        assert name in ['G', 'D']
        var_scope = 'generator' if name == 'G' else 'discriminator'
        with tf.name_scope('train_{}'.format(name)):
            opt = tf.train.AdamOptimizer(beta1=0.5,
                                         learning_rate=self.lr)
            loss = self.get_loss(name)
            var_list = tf.trainable_variables(scope=var_scope)
            grads = tf.gradients(loss, var_list)
            # grads = opt.compute_gradients(loss)

            return opt.apply_gradients(zip(grads, var_list),
                                       name='train_{}'.format(name))

        # grads = tf.gradients(loss, var_list)
        # [tf.summary.histogram('gradient/' + var.name, grad, 
        #       collections = [tf.GraphKeys.SUMMARIES]) for grad, var in zip(grads, var_list)]
        # grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
        # opt = tf.train.AdamOptimizer(cur_lr)
        # # opt = tf.train.RMSPropOptimizer(learning_rate=cur_lr)
        # train_op = opt.apply_gradients(zip(grads, var_list),
        #                                global_step=global_step)
            
    def get_loss(self, name):
        assert name in ['G', 'D']
        try:
            return self.g_loss if name == 'G' else self.d_loss
        except AttributeError:
            self.g_loss = self._g_loss()
            self.d_loss = self._d_loss()
            return self.g_loss if name == 'G' else self.d_loss

    def get_summary(self, key):
        assert key in ['train', 'test']
        return tf.summary.merge_all(key=key)

    def setup_summary(self):
        tf.summary.image(
            'generate',
            tf.cast(self.layers['fake'], tf.float32),
            collections=['train'])
        tf.summary.image(
            'label',
            tf.cast(self.label, tf.float32),
            collections=['train'])
        tf.summary.image(
            'input',
            tf.cast(self.image, tf.float32),
            collections=['train'])

        tf.summary.image(
            'test_generate',
            tf.cast(self.layers['fake'], tf.float32),
            collections=['test'])
        tf.summary.image(
            'test_input',
            tf.cast(self.image, tf.float32),
            collections=['test'])
         
    def _d_loss(self):
        with tf.name_scope('d_loss'):
            logits_d_real = self.layers['d_real']
            logits_d_fake = self.layers['d_fake']
            loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(logits_d_real),
                logits=logits_d_real,
                name='d_real')
            loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(logits_d_fake),
                logits=logits_d_fake,
                name='d_fake')

        return (tf.reduce_mean(loss_fake) + tf.reduce_mean(loss_real)) / 2.0

    def _g_loss(self):
        with tf.name_scope('g_loss'):
            logits_d_fake = self.layers['d_fake']
            loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(logits_d_fake),
                logits=logits_d_fake,
                name='g_fake')
            l1_loss = tf.reduce_mean(
                tf.abs(self.label - self.layers['fake']), name='l1')
            # l1_loss = tf.losses.absolute_difference(
            #     labels=self.label,
            #     predictions=self.layers['fake'],
            #     weights=1.0,
            #     scope='l1',
            #     # loss_collection=tf.GraphKeys.LOSSES,
            #     # reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
            #     )
        return tf.reduce_mean(loss_fake) + 100.0 * l1_loss

    def _generator(self, x_inputs):
        with tf.variable_scope('generator') as scope:
            self.layers['cur_input'] = x_inputs
            with tf.variable_scope('encoder'):
                arg_scope = tf.contrib.framework.arg_scope
                with arg_scope([L.conv_bn_leaky],
                               filter_size=4,
                               stride=2,
                               leak=0.2,
                               layer_dict=self.layers):
                    L.conv_bn_leaky(out_dim=64, name='encoder_1', bn=False)
                    L.conv_bn_leaky(out_dim=128, name='encoder_2', bn=True)
                    L.conv_bn_leaky(out_dim=256, name='encoder_3', bn=True)
                    L.conv_bn_leaky(out_dim=512, name='encoder_4', bn=True)
                    L.conv_bn_leaky(out_dim=512, name='encoder_5', bn=True)
                    L.conv_bn_leaky(out_dim=512, name='encoder_6', bn=True)
                    L.conv_bn_leaky(out_dim=512, name='encoder_7', bn=True)
                    L.conv_bn_leaky(out_dim=512, name='encoder_8', bn=True)

            with tf.variable_scope('decoder'):
                arg_scope = tf.contrib.framework.arg_scope
                with arg_scope([L.deconv_bn_drop_relu],
                                filter_size=4,
                                stride=2,
                                keep_prob=self.keep_prob,
                                layer_dict=self.layers):
                    L.deconv_bn_drop_relu(out_dim=512, name='decoder_1', fusion_id=7, bn=False)
                    L.deconv_bn_drop_relu(out_dim=512, name='decoder_2', fusion_id=6, bn=True)
                    L.deconv_bn_drop_relu(out_dim=512, name='decoder_3', fusion_id=5, bn=True)
                with arg_scope([L.deconv_bn_drop_relu],
                                filter_size=4,
                                stride=2,
                                keep_prob=1,
                                layer_dict=self.layers):
                    L.deconv_bn_drop_relu(out_dim=512, name='decoder_4', fusion_id=4, bn=True)
                    L.deconv_bn_drop_relu(out_dim=256, name='decoder_5', fusion_id=3, bn=True)
                    L.deconv_bn_drop_relu(out_dim=128, name='decoder_6', fusion_id=2, bn=True)
                    L.deconv_bn_drop_relu(out_dim=64, name='decoder_7', fusion_id=1, bn=True)

            self.layers['cur_input'] = tf.tanh(L.transpose_conv(
                self.layers['cur_input'],
                filter_size=4,
                out_dim=self._y_n_c,
                name='dconv',
                stride=2,
                # init_w=init_w,
                # init_b=init_b,
                nl=tf.identity))

            return self.layers['cur_input']

    def _discriminator(self, x_inputs, y_inputs):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
            self.layers['cur_input'] = tf.concat((x_inputs, y_inputs), axis=-1)

            arg_scope = tf.contrib.framework.arg_scope
            with arg_scope([L.conv_bn_leaky],
                           filter_size=4,
                           stride=2,
                           leak=0.2,
                           layer_dict=self.layers):
                L.conv_bn_leaky(out_dim=64, name='discrimin_1', bn=False)
                L.conv_bn_leaky(out_dim=128, name='discrimin_2', bn=True)
                L.conv_bn_leaky(out_dim=256, name='discrimin_3', bn=True)
                L.conv_bn_leaky(out_dim=512, name='discrimin_4', bn=True)

                # self.layers['cur_input'] = tf.reshape(
                #     self.layers['cur_input'],
                #     [-1, 16, 16, 512])
                # patch_avg = layers.fc(
                #     self.layers['cur_input'],
                #     out_dim=1,
                #     name='fc',
                #     nl=tf.identity, 
                #     init_w=tf.random_normal_initializer(stddev=INIT_W_STD),
                #     init_b=tf.zeros_initializer(),
                #     )

                fc_out = layers.conv(
                    self.layers['cur_input'],
                    filter_size=4,
                    out_dim=1,
                    name='fc',
                    stride=1,
                    init_w = tf.random_normal_initializer(stddev=INIT_W_STD),
                    init_b = tf.zeros_initializer(),
                    nl=tf.identity)
                # patch_avg = layers.global_avg_pool(fc_out, name='gap')

                self.layers['discrimin_logits'] = fc_out
                self.layers['cur_input'] = fc_out
                return fc_out
