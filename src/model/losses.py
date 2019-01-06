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
        # obj_mask = tf.cast(mask, tf.int32)
        # pos_obj_mask = obj_mask
        # neg_obj_mask = tf.logical_and(1 - obj_mask)
        # [bsize, len, 1]
        obj_loss = tf.nn.weighted_cross_entropy_with_logits(
            targets=label, logits=logits, pos_weight=1., name='obj_loss')
        # ignore the objectness loss with sign == 1
        masked_obj_loss = apply_mask(obj_loss, ignore_mask, val=1)
        return tf.reduce_sum(masked_obj_loss) # [n_non_ignore]

def bboxes_loss(label, pred_t_coord, obj_mask, name='bboxes_loss'):
    with tf.name_scope(name):
        # [bsize, len, 1]
        bbox_xy, bbox_wh = tf.split(pred_t_coord, [2, 2], axis=-1)
        pred_t_coord = tf.concat([tf.nn.sigmoid(bbox_xy), bbox_wh], axis=-1)

        bbox_loss = tf.reduce_sum(tf.square(pred_t_coord - label), axis=-1, keepdims=True)
        # only count for target anchors
        masked_bbox_loss = apply_mask(bbox_loss, obj_mask, val=1)
        return tf.reduce_sum(masked_bbox_loss) # [n_obj]

def class_loss(label, pred_class, obj_mask, name='class_loss'):
    with tf.name_scope(name):
        # [bsize, len, n_class]
        cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label, logits=pred_class)
        # [bsize, len, 1]
        cls_loss = tf.reduce_sum(cls_loss, axis=-1, keepdims=True)
        # only count for target anchors
        masked_cls_loss = apply_mask(cls_loss, obj_mask, val=1)
        return tf.reduce_sum(masked_cls_loss) # [n_obj]
