#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope
import lib.utils.utils as utils
# import tensorcv.models.layers as layers

INIT_W_STD = 0.02

@add_arg_scope
def conv_bn_leaky(
    filter_size, out_dim, stride, leak, layer_dict, name, bn=True):
    with tf.variable_scope(name):
        inputs = layer_dict['cur_input']
        conv_out = conv(
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
            bn_conv_out = batch_norm(conv_out, train=True, name='bn')
        else:
            bn_conv_out = conv_out

        leak_bn_conv_out = leaky_relu(
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
            bn_deconv_out = batch_norm(deconv_out, train=True, name='bn')
        else:
            bn_deconv_out = deconv_out
        drop_out_bn = dropout(
            bn_deconv_out, keep_prob, is_training=True, name='dropout')

        if fusion_id is not None:
            layer_dict['cur_input'] = tf.concat(
              (drop_out_bn, layer_dict['encoder_{}'.format(fusion_id)]),
              axis=-1)
        else:
            layer_dict['cur_input'] = drop_out_bn

        layer_dict['cur_input'] = tf.nn.relu(layer_dict['cur_input'])

        return layer_dict['cur_input']

def conv(x, 
         filter_size,
         out_dim, 
         stride=1, 
         padding='SAME',
         nl=tf.identity,
         init_w=None,
         init_b=None,
         use_bias=True,
         wd=None,
         trainable=True,
         name='conv'):

    in_dim = int(x.shape[-1])
    assert in_dim is not None,\
    'Number of input channel cannot be None!'

    filter_shape = utils.get_shape2D(filter_size) + [in_dim, out_dim]
    strid_shape = utils.get_shape4D(stride)

    padding = padding.upper()

    convolve = lambda i, k: tf.nn.conv2d(i, k, strid_shape, padding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights',
                                  filter_shape,
                                  initializer=init_w,
                                  trainable=trainable)

        out = convolve(x, weights)

        if use_bias:
            biases = tf.get_variable('biases',
                                     [out_dim],
                                     initializer=init_b,
                                     trainable=trainable)

            out = tf.nn.bias_add(out, biases)
        
        output = nl(out, name = 'output')
        return output

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

    stride = utils.get_shape4D(stride)

    in_dim = x.get_shape().as_list()[-1]

    # TODO other ways to determine the output shape 
    x_shape = tf.shape(x)
    # assume output shape is input_shape*stride
    if out_shape is None:
        out_shape = tf.stack([x_shape[0],
                              tf.multiply(x_shape[1], stride[1]), 
                              tf.multiply(x_shape[2], stride[2]),
                              out_dim])        

    filter_shape = utils.get_shape2D(filter_size) + [out_dim, in_dim]

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

def batch_norm(x, train=True, name='bn'):
    """ 
    batch normal 
    Args:
        x (tf.tensor): a tensor 
        name (str): name scope
        train (bool): whether training or not
    Returns:
        tf.tensor with name 'name'
    """
    return tf.contrib.layers.batch_norm(
        x, decay=0.9, updates_collections=None,
        epsilon=1e-5, scale=False, is_training=train, scope=name)

def leaky_relu(x, leak=0.2, name='LeakyRelu'):
    """ 
    leaky_relu 
        Allow a small non-zero gradient when the unit is not active
    Args:
        x (tf.tensor): a tensor 
        leak (float): Default to 0.2
    Returns:
        tf.tensor with name 'name'
    """
    return tf.maximum(x, leak*x, name=name)

def dropout(x, keep_prob, is_training, name='dropout'):
    """ 
    Dropout 
    Args:
        x (tf.tensor): a tensor 
        keep_prob (float): keep prbability of dropout
        is_training (bool): whether training or not
        name (str): name scope
    Returns:
        tf.tensor with name 'name'
    """

    # tf.nn.dropout does not have 'is_training' argument
    # return tf.nn.dropout(x, keep_prob)
    return tf.layers.dropout(
        x, rate=1 - keep_prob, training=is_training, name=name)