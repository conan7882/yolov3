#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: losses.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf


def apply_mask(input_matrix, mask, val=1, name='apply_mask'):
    with tf.name_scope(name):
        mask = tf.cast(mask, tf.int32)
        return tf.dynamic_partition(input_matrix, mask, 2)[val]

def objectness_loss(label, logits, ignore_mask, name='obj_loss'):
    with tf.name_scope(name):
        # [bsize, len, 1]
        obj_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label, logits=logits, name='obj_loss')
        # ignore the objectness loss with sign == 1
        masked_obj_loss = apply_mask(obj_loss, ignore_mask, val=0)
        return masked_obj_loss # [len]

def bboxes_loss(label, pred_t_coord, obj_mask, name='bboxes_loss'):
    with tf.name_scope(name):
        # [bsize, len, 1]
        bbox_loss = tf.reduce_sum(tf.square(pred_t_coord - label), axis=-1, keepdims=True)
        # only count for target anchors
        masked_bbox_loss = apply_mask(bbox_loss, obj_mask, val=1)
        return masked_bbox_loss # [len]

def class_loss(label, pred_class, obj_mask, name='class_loss'):
    with tf.name_scope(name):
        # [bsize, len, n_class]
        cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label, logits=pred_class)
        # [bsize, len, 1]
        cls_loss = tf.reduce_sum(cls_loss, axis=-1, keepdims=True)
        # only count for target anchors
        masked_cls_loss = apply_mask(cls_loss, obj_mask, val=1)
        return masked_cls_loss # [len]