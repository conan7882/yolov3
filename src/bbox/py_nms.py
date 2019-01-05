#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: py_nms.py
# Author: Qian Ge <geqian1001@gmail.com>
# reference:
# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

import numpy as np


def non_max_suppression(bbox_list, prob_list, overlap_thr, box_type='xyxy'):
    # print('bbox_list: {}'.format(bbox_list))
    # print('size: {}'.format(len(bbox_list)))
    if len(bbox_list) == 0:
        return bbox_list, prob_list

    if box_type == 'cxywh':
        bbox_list = np.stack(
            [bbox_list[:, 0] - bbox_list[:, 2] / 2,
             bbox_list[:, 1] - bbox_list[:, 3] / 2,
             bbox_list[:, 0] + bbox_list[:, 2] / 2,
             bbox_list[:, 1] + bbox_list[:, 3] / 2], axis=-1)

    prob_list = np.reshape(prob_list, [-1])
    assert len(bbox_list) == len(prob_list)

    order_idx = np.argsort(prob_list)

    # area_list = (bbox_list[:, 2] - bbox_list[:, 0] + 1) * (bbox_list[:, 3] - bbox_list[:, 1] + 1)
    x1 = bbox_list[:, 0]
    x2 = bbox_list[:, 2]
    y1 = bbox_list[:, 1]
    y2 = bbox_list[:, 3]
    area_list = (x2 - x1 + 1) * (y2 - y1 + 1)

    pick_idx = []
    while len(order_idx) > 0:
        cur_idx = order_idx[-1]
        pick_idx.append(cur_idx)
        # cur_bbox = np.array([bbox_list[cur_idx]])

        other_area = area_list[order_idx[:-1]]
        cur_area = area_list[cur_idx]

        xx1 = np.maximum(x1[cur_idx], x1[order_idx[:-1]])
        yy1 = np.maximum(y1[cur_idx], y1[order_idx[:-1]])
        xx2 = np.minimum(x2[cur_idx], x2[order_idx[:-1]])
        yy2 = np.minimum(y2[cur_idx], y2[order_idx[:-1]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        i_area = w * h

        overlap_list = i_area / (area_list[cur_idx] + area_list[order_idx[:-1]] - i_area)

        order_idx = np.delete(order_idx, np.concatenate(
                ([len(order_idx) - 1], np.where(overlap_list >= overlap_thr)[0])))
        # print(order_idx)

    return bbox_list[pick_idx], prob_list[pick_idx]
