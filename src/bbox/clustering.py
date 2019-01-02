#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: clustering.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import src.bbox.bboxtool as bboxtool


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
            iou = bboxtool.bbox_list_IOU(self._bbox_list, centroid, align=True)
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

        iou = bboxtool.bbox_list_IOU(self._bbox_list, self._centroid, align=True)
        max_iou = np.amax(iou, axis=-1)
        return np.mean(max_iou)





