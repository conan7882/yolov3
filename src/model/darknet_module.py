#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: darknet_module.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import src.model.layers as L


def darknet_padding(inputs, fix_pad=1):
    """ Zeros padding with fixed padding size

        Args:
            inputs (tensor): input feature map [bsize, h, w, channel]
            fix_pad (int): padding size for one side

    """
    return tf.pad(
        inputs,
        [[0, 0], [fix_pad, fix_pad], [fix_pad, fix_pad], [0, 0]],
        "CONSTANT")

def leaky(x):
    """ leaky ReLU with alpha = 0.1 """
    return L.leaky_relu(x, leak=0.1)

def classification_fc(inputs, 
                      wd=0,
                      init_w=None,
                      n_class=1000, 
                      trainable=False, 
                      is_training=True,
                      pretrained_dict=None,
                      init_b=tf.zeros_initializer(),
                      name='classification_fc'):

    """ Fully connected layer for classification implemented as 1x1 conv layer

        Args:
            inputs (tensor): input feature maps
            n_class (int): number of classes for classification
            pretrained_dict (dict): dictionary of pre-trained weights with keys
                the same as the layer names
            init_w, init_b: tf initializer for weights and bias
            wd (float): weight decay
            trainable (bool): whether train the subnet or not
            is_training (bool): is used for a training model or not

        Returns:
            logits for classifcation with shape [bsize, n_class]
    """
    layer_dict = {}
    layer_dict['cur_input'] = inputs

    with tf.variable_scope(name):
        L.global_avg_pool(layer_dict, keepdims=True)
        L.conv(filter_size=1,
               out_dim=n_class,
               layer_dict=layer_dict, 
               bn=False,
               wd=wd,
               init_w=init_w,
               init_b=init_b,
               nl=tf.identity,
               trainable=trainable,
               is_training=is_training,
               pretrained_dict=pretrained_dict,
               name='conv_fc_1')

        layer_dict['cur_input'] = tf.squeeze(layer_dict['cur_input'], [1, 2])
        return layer_dict['cur_input']

def darkent53_conv(inputs, 
                   wd=0, bn=True, 
                   trainable=False, 
                   is_training=True,
                   pretrained_dict=None,
                   init_w=None,
                   init_b=tf.zeros_initializer(),
                   name='darkent53_conv'):

    """ Conv layers for Darknet53

        Args:
            inputs (tensor): input feature maps
            pretrained_dict (dict): dictionary of pre-trained weights with keys
                the same as the layer names
            init_w, init_b: tf initializer for weights and bias
            bn (bool): use batch normalization or not
            wd (float): weight decay
            trainable (bool): whether train the subnet or not
            is_training (bool): is used for a training model or not

        Returns:
            feature maps with stride 32, 16 and 8
    """

    def conv_layer(filter_size, out_dim, stride, custom_padding=None):
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.conv],
                       layer_dict=layer_dict,
                       wd=wd, bn=bn, 
                       nl=leaky, 
                       init_w=init_w,
                       use_bias=False, 
                       trainable=trainable,
                       is_training=is_training, 
                       custom_padding=custom_padding,
                       pretrained_dict=pretrained_dict,):

            L.conv(filter_size=filter_size, out_dim=out_dim, stride=stride,
                   name='conv_{}'.format(layer_dict['layer_id']))
            layer_dict['layer_id'] += 1

            return layer_dict['cur_input']    

    def res_block(n_block, out_dim):
        conv_layer(filter_size=3, out_dim=out_dim, stride=2,
                   custom_padding=darknet_padding)
        res_input = layer_dict['cur_input']
        for i in range(n_block):
            conv_layer(filter_size=1, out_dim=out_dim // 2, stride=1)
            conv_layer(filter_size=3, out_dim=out_dim, stride=1)
            layer_dict['cur_input'] += res_input
            res_input = layer_dict['cur_input']
        return layer_dict['cur_input']

    layer_dict = {}
    layer_dict['cur_input'] = inputs

    with tf.variable_scope(name):
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.conv], 
                       layer_dict=layer_dict,
                       bn=bn, wd=wd, 
                       nl=leaky, 
                       init_w=init_w,
                       use_bias=False, 
                       trainable=trainable,
                       is_training=is_training,
                       pretrained_dict=pretrained_dict,):

            layer_dict['layer_id'] = 1
            conv_layer(filter_size=3, out_dim=32, stride=1)
            res_block(n_block=1, out_dim=64)
            res_block(n_block=2, out_dim=128)
            res_block(n_block=8, out_dim=256)
            route_2 = layer_dict['cur_input']
            res_block(n_block=8, out_dim=512)
            route_1 = layer_dict['cur_input']
            res_block(n_block=4, out_dim=1024)

            return layer_dict['cur_input'], route_1, route_2
