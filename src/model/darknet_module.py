#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: darknet_module.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import src.model.layers as L


def darknet_padding(inputs, fix_pad=1):
    return tf.pad(
        inputs,
        [[0, 0], [fix_pad, fix_pad], [fix_pad, fix_pad], [0, 0]],
        "CONSTANT")

def leaky(x):
    return L.leaky_relu(x, leak=0.1)

def classification_fc(inputs, n_class=1000, pretrained_dict=None, 
                      init_w=None, init_b=tf.zeros_initializer(),
                      wd=0, trainable=False, is_training=True,
                      name='classification_fc'):
    layer_dict = {}
    layer_dict['cur_input'] = inputs

    with tf.variable_scope(name):
        L.global_avg_pool(layer_dict, keepdims=True)
        L.conv(filter_size=1, out_dim=n_class, layer_dict=layer_dict, 
               pretrained_dict=pretrained_dict, 
               bn=False, nl=tf.identity, init_w=init_w, init_b=init_b,
               trainable=trainable, is_training=is_training,
               wd=wd, name='conv_fc_1')

        layer_dict['cur_input'] = tf.squeeze(layer_dict['cur_input'], [1, 2])
        return layer_dict['cur_input']

def darkent53_conv(inputs, pretrained_dict=None,
                   init_w=None, init_b=tf.zeros_initializer(),
                   bn=True, wd=0, trainable=False, is_training=True,
                   name='darkent53_conv'):

    def conv_layer(filter_size, out_dim, stride, custom_padding=None):
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.conv], layer_dict=layer_dict, pretrained_dict=pretrained_dict,
                       bn=bn, nl=leaky, init_w=init_w, wd=wd, is_training=is_training,
                       use_bias=False, trainable=trainable, custom_padding=custom_padding):
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
        with arg_scope([L.conv], layer_dict=layer_dict, pretrained_dict=pretrained_dict,
                       bn=bn, nl=leaky, init_w=init_w, wd=wd, is_training=is_training,
                       use_bias=False, trainable=trainable):
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
