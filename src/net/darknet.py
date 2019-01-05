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
    """ class of Darknet53 only  for inference """
    def __init__(self, n_channel, n_class, pre_trained_path, trainable=False):
        """
            Args:
                n_channel (int): number of channel of input image
                n_class (int): number of classes for classification
                pre_trained_path (str): path of pre-trained model (.npy file)
                trainable (bool): whether train this net or not (training is not implemented)
        """
        self._n_channel = n_channel
        self._n_class = n_class
        self._trainable = trainable

        # load pre-trained model
        self._pretrained_dict = None
        if pre_trained_path:
            self._pretrained_dict = np.load(
                pre_trained_path, encoding='latin1').item()

        self.layers = {}

    def _create_valid_input(self):
        """ Create inputs of the model. 
            Input images should be scale to [0, 1] before fed into the model.

        """
        self.image = tf.placeholder(
            tf.float32, [None, None, None, self._n_channel], name='image')

    def create_valid_model(self):
        """ Create model for inference. 
            Get top 5 prediction for each image.
        """
        self.set_is_training(is_training=False)
        self._create_valid_input()
        self.layers['logits'] = self._create_model(self.image)
        # get top 5 prediction for each image
        self.layers['top_5'] = tf.nn.top_k(
                tf.nn.softmax(self.layers['logits']), k=5, sorted=True)

    def _create_model(self, inputs):
        """ Create darknet53 classification model
            Return logits
        """
        with tf.variable_scope('DarkNet53', reuse=tf.AUTO_REUSE):
            conv_out, _, _ = module.darkent53_conv(
                inputs,
                init_w=INIT_W,
                bn=True, wd=0, 
                trainable=self._trainable, 
                is_training=self.is_training,
                pretrained_dict=self._pretrained_dict,
                name='darkent53_conv')

            logits = module.classification_fc(
                conv_out,
                n_class=self._n_class,
                init_w=INIT_W, wd=0,
                trainable=self._trainable,
                is_training=self.is_training,
                pretrained_dict=self._pretrained_dict,
                name='darknet53_fc')

            return logits
