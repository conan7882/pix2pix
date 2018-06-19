#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

import tensorcv.models.layers as layers

INIT_W_STD = 0.02

@add_arg_scope
def conv_bn_leaky(
    filter_size, out_dim, stride, leak, layer_dict, name, bn=True):
    with tf.variable_scope(name):
        inputs = layer_dict['cur_input']
        conv_out = layers.conv(
            inputs,
            filter_size=filter_size,
            out_dim=out_dim,
            name='conv',
            stride=stride,
            padding='SAME',
            init_w = tf.random_normal_initializer(stddev=INIT_W_STD),
            init_b=tf.zeros_initializer(),
            nl=tf.identity)
        if bn == True:
            bn_conv_out = layers.batch_norm(conv_out, train=True, name='bn')
        else:
            bn_conv_out = conv_out

        leak_bn_conv_out = layers.leaky_relu(
            bn_conv_out, leak=leak, name='LeakyRelu')
        layer_dict[name] = leak_bn_conv_out
        layer_dict['cur_input'] = leak_bn_conv_out
        return leak_bn_conv_out

@add_arg_scope
def deconv_bn_drop_relu(
    filter_size, out_dim, stride, keep_prob, layer_dict, name, fusion_id=None, bn=True):
    with tf.variable_scope(name):
        inputs = layer_dict['cur_input']
        deconv_out = transpose_conv(
            inputs,
            filter_size=filter_size,
            out_dim=out_dim,
            stride=stride,
            padding='SAME',
            name='dconv')
        if bn == True:
            bn_deconv_out = layers.batch_norm(deconv_out, train=True, name='bn')
        else:
            bn_deconv_out = deconv_out
        drop_out_bn = layers.dropout(
            bn_deconv_out, keep_prob, is_training=True, name='dropout')

        if fusion_id is not None:
            layer_dict['cur_input'] = tf.concat(
              (drop_out_bn, layer_dict['encoder_{}'.format(fusion_id)]),
              axis=-1)
        else:
            layer_dict['cur_input'] = drop_out_bn

        layer_dict['cur_input'] = tf.nn.relu(layer_dict['cur_input'])

        return layer_dict['cur_input']

def transpose_conv(x,
                   filter_size,
                   out_dim,
                   out_shape=None,
                   use_bias=True,
                   stride=2,
                   padding='SAME',
                   trainable=True,
                   nl=tf.identity,
                   name='dconv'):

    stride = layers.get_shape4D(stride)

    in_dim = x.get_shape().as_list()[-1]

    # TODO other ways to determine the output shape 
    x_shape = tf.shape(x)
    # assume output shape is input_shape*stride
    if out_shape is None:
        out_shape = tf.stack([x_shape[0],
                              tf.multiply(x_shape[1], stride[1]), 
                              tf.multiply(x_shape[2], stride[2]),
                              out_dim])        

    filter_shape = layers.get_shape2D(filter_size) + [out_dim, in_dim]

    with tf.variable_scope(name) as scope:
        init_w = tf.random_normal_initializer(stddev=INIT_W_STD)
        init_b = tf.zeros_initializer()
        weights = tf.get_variable('weights',
                                  filter_shape,
                                  initializer=init_w,
                                  trainable=trainable)
        biases = tf.get_variable('biases',
                                 [out_dim],
                                 initializer=init_b,
                                 trainable=trainable)
        

        output = tf.nn.conv2d_transpose(x,
                                        weights, 
                                        output_shape=out_shape, 
                                        strides=stride, 
                                        padding=padding, 
                                        name=scope.name)

        output = tf.nn.bias_add(output, biases)
        output.set_shape([None, None, None, out_dim])
        output = nl(output, name='output')
        return output

