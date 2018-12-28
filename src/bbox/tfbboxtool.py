#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tfbboxtool.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

def cxywh2xyxy(bbox_list):
    xyxy_bbox = tf.stack(
        [bbox_list[:, 0] - bbox_list[:, 2] / 2,
         bbox_list[:, 1] - bbox_list[:, 3] / 2,
         bbox_list[:, 0] + bbox_list[:, 2] / 2,
         bbox_list[:, 1] + bbox_list[:, 3] / 2,], axis=-1)
    return xyxy_bbox

def rescale_bbox(bbox_list, from_shape, to_shape):
    # from_shape, to_shape [h, w]
    # bbox_list[i] [x, y, x, y] or [cx, xy, w, h]
    rescale_bbox = tf.stack(
        [bbox_list[:, 0] / from_shape[1] * to_shape[1],
         bbox_list[:, 1] / from_shape[0] * to_shape[0],
         bbox_list[:, 2] / from_shape[1] * to_shape[1],
         bbox_list[:, 3] / from_shape[0] * to_shape[0]], axis=-1)
    return rescale_bbox