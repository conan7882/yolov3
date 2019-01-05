#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: np_eval.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import src.utils.utils as utils
import src.bbox.bboxtool as bboxtool


def mAP(pred_bboxes,
        pred_classes,
        pred_conf,
        gt_bboxes,
        gt_classes,
        IoU_thr,
        pred_im_size,
        gt_im_size):
    """ evaluate object detection performance using mAP

        Args:
            pred_bboxes, gt_bboxes (list): prediction and groundtruth bounding box
                with size [n_bbox, 4] with format xyxy
            pred_classes, gt_classes (list): prediction and groundtruth class labels
                corresponding the the pred_bboxes and gt_bboxes
            pred_conf (list): confidence score of each prediction
            IoU_thr (float): IoU threshold to determine the correct detection
            pred_im_size, gt_im_size (int or list of two int): image size for prediction
                groundtruth bounding box

        Returns:
            a scalar value of mAP (float) 
    """
    # bbox xyxy

    pred_classes, gt_classes, pred_bboxes, gt_bboxes, pred_conf =\
        utils.to_nparray([pred_classes, gt_classes, pred_bboxes, gt_bboxes, pred_conf])
    # rescale bbox to the same scale
    pred_bboxes = bboxtool.rescale_bbox(pred_bboxes, pred_im_size, gt_im_size)

    total_classes = set(pred_classes).union(set(gt_classes))
    recall_step = np.linspace(0,1,11)
    len_recall_step = len(recall_step)
    AP_classes = [0 for _ in range(len(total_classes))]
    for c_cnt, c_id in enumerate(total_classes):
        # get bbox for the current class only
        pred_id = np.where(pred_classes == c_id)[0]
        c_pred_bbox = pred_bboxes[pred_id]
        c_pred_conf = pred_conf[pred_id]

        gt_id = np.where(gt_classes == c_id)[0]
        c_gt_bbox = gt_bboxes[gt_id]
        n_gt = len(c_gt_bbox)

        # AP is 0 if this class does not in either prediction or gt
        if len(pred_id) == 0 or len(gt_id) == 0:
            AP_classes[c_cnt] = 0
            continue

        # get corrent detection based on IoUs between prediction and gt
        # IoU_mat [n_gt, n_pred]
        IoU_mat = bboxtool.bbox_list_IOU(c_gt_bbox, c_pred_bbox, align=False)
        det_gt_list = np.argmax(IoU_mat, axis=0)
        iou_list = IoU_mat[det_gt_list, np.arange(len(det_gt_list))]
        iou_list[np.where(iou_list < IoU_thr)] = 0
        
        # make table of IoU, prediction confidence and detected gt_id for
        # sorting the results based on prediction confidence
        det_table = np.stack((iou_list, c_pred_conf, det_gt_list), axis=-1)
        det_table = det_table[det_table[:, 1].argsort()[::-1]]

        # compute recall and precision for each confidence threshold
        recall_list = [0 for _ in range(len(iou_list))]
        precision_list = [0 for _ in range(len(iou_list))]
        prev_precision = 0.
        TP_id = (det_table[:,0] > 0)
        peak_list = []
        for i in range(len(iou_list)):
            recall_list[i] = len(set(det_gt_list[:i+1][TP_id[:i+1]])) / n_gt
            precision_list[i] = sum(det_table[:i+1,0] > 0) / (i + 1)
            if precision_list[i] < prev_precision:
                peak_list.append((prev_precision, recall_list[i - 1]))
            prev_precision = precision_list[i]
        peak_list.append((prev_precision, recall_list[-1]))

        # get max precision for each recall level
        max_precision = [0 for _ in range(len_recall_step)]
        peak_p = 0
        max_ = 0
        for idx, recall_ in enumerate(recall_step):
            while peak_p < len(peak_list) and peak_list[peak_p][1] <= recall_:
                max_ = max(max_, peak_list[peak_p][0])
                peak_p += 1
            max_precision[idx] = max_
            if peak_p < len(peak_list):
                max_ = peak_list[peak_p][0]
        max_precision[0] = max(max_precision)
        AP_classes[c_cnt] = np.mean(max_precision)

    return np.mean(AP_classes)
         



        