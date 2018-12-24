#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: darknet.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from src.net.base import BaseModel
import src.model.darknet_module as module

INIT_W = None

class DarkNet53(BaseModel):
    def __init__(self, n_channel, n_class, pre_trained_path, trainable=False):
        self._n_channel = n_channel
        self._n_class = n_class
        self._trainable = trainable

        self._pretrained_dict = None
        if pre_trained_path:
            self._pretrained_dict = np.load(
                pre_trained_path, encoding='latin1').item()

        self.layers = {}

    def _create_valid_input(self):
        self.image = tf.placeholder(
            tf.float32, [None, None, None, self._n_channel], name='image')
        # self.image = module.sub_rgb2bgr_mean(self.raw_image)
        self.label = tf.placeholder(tf.int64, [None], 'label')
        self.keep_prob = 1.

    def create_valid_model(self):
        self.set_is_training(is_training=False)
        self._create_valid_input()
        self.layers['logits'] = self._create_model(self.image)
        # self.layers['prob'] = tf.nn.softmax(self.layers['logits'], axis=-1)
        self.layers['top_5'] = tf.nn.top_k(
                tf.nn.softmax(self.layers['logits']), k=5, sorted=True)

    def _create_model(self, inputs):
        with tf.variable_scope('DarkNet53', reuse=tf.AUTO_REUSE):
            conv_out = module.darkent53_conv(
                inputs, pretrained_dict=self._pretrained_dict, init_w=INIT_W,
                bn=True, wd=0, trainable=self._trainable, is_training=self.is_training,
                name='darkent53_conv')
            logits = module.classification_fc(
                conv_out, n_class=self._n_class, pretrained_dict=self._pretrained_dict,
                init_w=INIT_W, wd=0, trainable=self._trainable, is_training=self.is_training,
                name='darknet53_fc')

            return logits
