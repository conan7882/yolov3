#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: clustering.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import src.bbox.bboxtool as bboxtool


def bbox_list_IOU(bbox_list_1, bbox_list_2, align=True):
    bbox_list_1 = np.array(bbox_list_1)
    bbox_list_2 = np.array(bbox_list_2)
    if len(bbox_list_1.shape) == 1:
        bbox_list_1 = [bbox_list_1]
    elif len(bbox_list_1.shape) > 2:
        raise ValueError('Incorrect shape of bbox_list_1')

    if len(bbox_list_2.shape) == 1:
        bbox_list_2 = [bbox_list_2]
    elif len(bbox_list_2.shape) > 2:
        raise ValueError('Incorrect shape of bbox_list_2')

    if align:
        transpose_sign = False
        if len(bbox_list_2) < len(bbox_list_1):
            bbox_list_1, bbox_list_2 = bbox_list_2, bbox_list_1
            transpose_sign = True

        h_list = bbox_list_2[:, 3] - bbox_list_2[:, 1]
        w_list = bbox_list_2[:, 2] - bbox_list_2[:, 0]
        area_list = np.multiply(h_list, w_list)

        iou_list = []
        for bbox in bbox_list_1:
            h = bbox[3] - bbox[1]
            w = bbox[2] - bbox[0]
            area = h * w

            inter_h = np.minimum(h, h_list)
            inter_w = np.minimum(w, w_list)
            inter_area = np.multiply(inter_h, inter_w)
            iou = inter_area / (area_list + area - inter_area)
            iou_list.append(iou)

        iou_list = np.array(iou_list)
        if transpose_sign:
            return iou_list.transpose()
        else:
            return iou_list


# box [xmin, ymin, xmax, ymax]
class BboxClustering(object):
    def __init__(self, bbox_list):
        self._bbox_list = np.array(bbox_list)

    def clustering(self, num_cluster, max_iter=100):
        n_bbox = len(self._bbox_list)
        # init centroid
        centroid = self._bbox_list[np.random.choice(
            n_bbox, num_cluster, replace=False)]

        prev_label = np.zeros(n_bbox)
        cnt = 0
        while True:
            cnt += 1
            iou = bbox_list_IOU(self._bbox_list, centroid, align=True)
            dist = 1 - iou
            label = np.argmin(dist, axis=-1)
            for k in range(num_cluster):
                mean_h = np.median(self._bbox_list[label == k, 3] - self._bbox_list[label == k, 1], axis=0)
                mean_w = np.median(self._bbox_list[label == k, 2] - self._bbox_list[label == k, 0], axis=0)
                centroid[k] = [0, 0, mean_w, mean_h]

            if (prev_label == label).all() or cnt > max_iter:
                break
            prev_label = label
            
        self._centroid = centroid
        self._label = label

        return centroid

    def mean_iou(self, num_cluster=None):
        if num_cluster is not None:
            self.clustering(num_cluster=num_cluster, max_iter=100)
        try:
            self._centroid
        except AttributeError:
            self.clustering(num_cluster=5, max_iter=100)

        iou = bbox_list_IOU(self._bbox_list, self._centroid, align=True)
        max_iou = np.amax(iou, axis=-1)
        return np.mean(max_iou)





