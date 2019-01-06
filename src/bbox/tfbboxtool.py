#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tfbboxtool.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

def yolotcoord2cxywh():
    pass

def cxywh2xyxy(bbox_list):
    xyxy_bbox = tf.stack(
        [bbox_list[..., 0] - bbox_list[..., 2] / 2,
         bbox_list[..., 1] - bbox_list[..., 3] / 2,
         bbox_list[..., 0] + bbox_list[..., 2] / 2,
         bbox_list[..., 1] + bbox_list[..., 3] / 2,], axis=-1)
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

def batch_bbox_IoU(bbox_list_1, bbox_list_2):
    """ Pairwise bbox IoU for batch

        Args:
            bbox_list_1 (list[list[float]]): Batch of bbox [bsize, n, 4].
                Format is cxywh (yolo prediction format).
            bbox_list_2 (list[list[float]]): Batch of bbox [bsize, m, 4]
                Format is xyxy (groundtruth format).

        Returns:
            Pairwise IoU matrix (float) with shape [bsize, n, m].
            Element (k, n, m) is the IoU between nth bbox in bbox_list_1
            and mth bbox_list_2 in kth sample in the batch.

    """
    # boox1
    area_1 = bbox_list_1[..., 2] * bbox_list_1[..., 3] #[bsize, n]
    area_1 = tf.expand_dims(area_1, axis=-1) #[bsize, n, 1]
    # convert cxywh to xyxy format
    bbox_list_1 = cxywh2xyxy(bbox_list_1)

    # bbox2
    h_2, w_2 = bbox_list_2[..., 3] - bbox_list_2[..., 1],\
               bbox_list_2[..., 2] - bbox_list_2[..., 0]
    area_2 = h_2 * w_2 #[bsize, m]
    area_2 = tf.expand_dims(area_2, axis=1) #[bsize, 1, m]

    area_sum = area_1 + area_2 # [bsize, n, m]

    # [bsize, n, m, 2]
    inter_min = tf.maximum(tf.expand_dims(bbox_list_1[..., :2], axis=2),
                           tf.expand_dims(bbox_list_2[..., :2], axis=1))
    inter_max = tf.minimum(tf.expand_dims(bbox_list_1[..., 2:], axis=2),
                           tf.expand_dims(bbox_list_2[..., 2:], axis=1))
    # [bsize, n, m]
    inter_w = tf.maximum(inter_max[..., 0] - inter_min[..., 0], 0.)
    inter_h = tf.maximum(inter_max[..., 1] - inter_min[..., 1], 0.)
    inter_area = inter_w * inter_h
    iou = inter_area / (area_sum - inter_area)
    return iou



