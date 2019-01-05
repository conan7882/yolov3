#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cluster_bbox.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import argparse
import platform
import numpy as np

import sys
sys.path.append('../')
from src.dataflow.voc import get_voc_bbox
from src.bbox.clustering import BboxClustering


def plot_avgiou_vs_k(xml_dir, k_range):
    bbox_list = get_voc_bbox(xml_dir)
    kmeans_bbox = BboxClustering(bbox_list)
    iou_list = []
    for k in k_range:
        re = kmeans_bbox.mean_iou(num_cluster=k)
        iou_list.append(re)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(k_range, iou_list, 'o-')
    plt.show()

def kmeans_clustering(xml_dir, k):
    bbox_list = get_voc_bbox(xml_dir)
    kmeans_bbox = BboxClustering(bbox_list)
    centroid = kmeans_bbox.clustering(num_cluster=k, max_iter=100)
    avg_iou = kmeans_bbox.mean_iou()
    print(centroid)
    print(avg_iou)
    def bbox_area(bbox): 
        return bbox[2] * bbox[3]
    sorted_centroid = sorted(centroid, key=bbox_area)
    print(list(np.reshape([[b[2], b[3]] for b in sorted_centroid], -1)))

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--plot', action='store_true',
                        help='Plot average IOU vs number of cluster')
    parser.add_argument('--cluster', action='store_true',
                        help='Clustering bounding boxes')
    parser.add_argument('--k', type=int, default=5, help='Number of clusters')
    
    return parser.parse_args()


if __name__ == '__main__':
    if platform.node() == 'arostitan':
        raise ValueError('Data path does not setup on this platform!')
    elif platform.node() == 'aros04':
        xml_dir = 'E:/Dataset/VOCdevkit/VOC2007/Annotations/'
    else:
        xml_dir = '/Users/gq/workspace/Dataset/VOCdevkit/VOC2007/Annotations/'

    FLAGS = get_args()

    if FLAGS.plot:
        plot_avgiou_vs_k(xml_dir, range(1, 15))
    if FLAGS.cluster:
        kmeans_clustering(xml_dir, FLAGS.k)
