#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: detection_bbox.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import src.bbox.bboxtool as bboxtool
from src.bbox.py_nms import non_max_suppression


def detection(bbox_para, bbox_score, n_class, obj_score_thr, nms_iou_thr, label_dict,
              image_shape, rescale_shape, box_type='cxywh'):
    if not isinstance(rescale_shape, list):
        rescale_shape = [rescale_shape, rescale_shape]

    bbox_score = np.array(bbox_score)
    bbox_score = np.reshape(bbox_score, (-1, n_class))
    bbox_para = np.reshape(bbox_para, (-1, 4))

    det_list = []
    box_list = []
    score_list = []
    for score, box in zip(bbox_score, bbox_para):
        cur_det = np.where(score >= obj_score_thr)[0]
        if len(cur_det) > 0:
            det_list.extend(cur_det)
            score_list.extend(score[cur_det])
            box_list.extend([box for _ in range(len(cur_det))])

    box_list = np.array(box_list)
    score_list = np.array(score_list)

    nms_boxes = []
    nms_scores = []
    nms_label_names = []
    nms_label_ids = []
    det_obj = set(det_list)
    for obj in det_obj:
        # print(len(det_list), len(score_list), len(box_list))
        obj_ids = np.where(det_list == obj)[0]
        score = score_list[obj_ids]
        boxes = box_list[obj_ids]
        obj_label = label_dict[obj]

        n_boxes, n_score = non_max_suppression(
            boxes, score, overlap_thr=nms_iou_thr, box_type=box_type)
        nms_boxes.extend(n_boxes)
        nms_scores.extend(n_score)
        nms_label_names.extend([obj_label for _ in range(len(n_score))])
        nms_label_ids.extend([obj for _ in range(len(n_score))])
    if len(nms_label_names) == 0:
        return [], [], [], []
    nms_boxes = np.array(nms_boxes)
    nms_label_names = np.array(nms_label_names)
    nms_label_ids = np.array(nms_label_ids)
    rescale_shape = np.array(rescale_shape)
    image_shape = np.array(image_shape)
    nms_scores = np.array(nms_scores)

    nms_boxes = bboxtool.rescale_bbox(nms_boxes, rescale_shape, image_shape)
    return nms_boxes, nms_scores, nms_label_names, nms_label_ids
