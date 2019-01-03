#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bboxgt.py
# Author: Qian Ge <geqian1001@gmail.com>

import copy
import numpy as np
import src.bbox.bboxtool as bboxtool
from src.utils.dataflow import vec2onehot


class TargetAnchor(object):
    def __init__(self, rescale_shape_list, stride_list, prior_list, n_class, ignore_thr=0.5):
        self._n_class = n_class
        self._yolo_single_out_dim = 1 + 4 + 1 + n_class # 1: obj 4: bbox 1: obj_ignore
        self._stride_list = stride_list
        self._ignore_thr = ignore_thr

        self.init_anchors_dict = {}
        self.init_gt_mask_dict = {}
        if isinstance(rescale_shape_list, int):
            rescale_shape_list = [rescale_shape_list]
        for rescale_shape in rescale_shape_list:
            if isinstance(rescale_shape, int):
                rescale_shape = [rescale_shape, rescale_shape]
            else:
                assert len(rescale_shape) == 2 and rescale_shape[0] == rescale_shape[1]
            
            gt_mask = [[] for _ in range(len(stride_list))]
            for scale_id, (stride, anchor) in enumerate(zip(stride_list, prior_list)):
                for prior_id in range(len(anchor)):
                    gt_mask[scale_id].append(np.zeros((int(rescale_shape[0] / stride),
                                                       int(rescale_shape[0] / stride),
                                                       self._yolo_single_out_dim)))
            self.init_gt_mask_dict[rescale_shape[0]] = np.array(gt_mask)

            ind_list = []
            anchor_list = []
            sub2ind_dict = {}
            anchor_cnt = 0
            for scale_id, (stride, anchor) in enumerate(zip(stride_list, prior_list)):
                cols, rows = int(rescale_shape[0] / stride), int(rescale_shape[1] / stride)
                for row_id in range(rows):
                    for col_id in range(cols):
                        for prior_id, cur_anchor in enumerate(anchor):
                            
                            cur_index = {'stride': stride, 'prior': cur_anchor,
                                         'row_id': row_id, 'col_id': col_id,
                                         'scale_id': scale_id, 'prior_id': prior_id,
                                         'anchor': cur_anchor}
                            # cur_index = (stride, cur_anchor, row_id, col_id)
                            offset_anchor = [stride * col_id - cur_anchor[0] / 2,
                                             stride * row_id - cur_anchor[1] / 2,
                                             stride * col_id + cur_anchor[0] / 2,
                                             stride * row_id + cur_anchor[1] / 2]
                            ind_list.append(cur_index)
                            anchor_list.append(offset_anchor)
                            sub2ind_dict.setdefault((scale_id, row_id, col_id), []).append(anchor_cnt)
                            # sub2ind_dict[(row_id, col_id)] = anchor_cnt
                            anchor_cnt += 1
            # print(rescale_shape)
            self.init_anchors_dict[rescale_shape[0]] = {'index': ind_list, 'anchors': anchor_list, 'sub2ind': sub2ind_dict}

    def get_yolo_target_anchor(self, gt_bbox_batch, im_shape_batch, rescale_shape, is_flatten=False):
        # gt_bbox_batch [bsize, n_gt, 4] xyxy
        # im_shape_batch [bsize, 2]
        if isinstance(rescale_shape, int):
            rescale_shape = [rescale_shape, rescale_shape]

        batch_gt_mask = []
        init_anchors = self.init_anchors_dict[rescale_shape[0]]
        anchor_list = init_anchors['anchors']
        ind_list = init_anchors['index']
        sub2ind = init_anchors['sub2ind']

        # target_anchor_batch = [[] for _ in range(len(gt_bbox_batch))]
        # b_id = -1
        batch_gt_mask = []
        for gt_bbox, im_shape in zip(gt_bbox_batch, im_shape_batch):
            # print(len(gt_bbox))
            # b_id += 1
            gt_mask = copy.deepcopy(self.init_gt_mask_dict[rescale_shape[0]])
            gt_bbox_para, gt_bbox_label = self._convert_gt(gt_bbox, im_shape, rescale_shape)
            one_hot_label = vec2onehot(gt_bbox_label, self._n_class)
            gt_cxy = np.stack(
                [(gt_bbox_para[:, 0] + gt_bbox_para[:, 2]) / 2,
                 (gt_bbox_para[:, 1] + gt_bbox_para[:, 3]) / 2], axis=-1)
            for gt_id, gt_bbox in enumerate(gt_bbox_para):
                candidate_anchor = []
                anchor_idx_list = []
                for scale_id, stride in enumerate(self._stride_list):
                    anchor_feat_cxy = gt_cxy[gt_id] // stride
                    gt_feat_cxy =  gt_cxy[gt_id] / stride
                    anchor_idx_list += sub2ind[(scale_id, anchor_feat_cxy[1], anchor_feat_cxy[0])]
                for anchor_idx in anchor_idx_list:
                    candidate_anchor.append(anchor_list[anchor_idx])

                iou_mat = bboxtool.bbox_list_IOU([gt_bbox], candidate_anchor, align=False)
                ignore_idx_list = np.where(iou_mat >= self._ignore_thr)[1]
                for ignore_idx in ignore_idx_list:
                    anchor_idx = anchor_idx_list[ignore_idx]
                    scale_id, prior_id, row_id, col_id, _, _ = self._get_anchor_property(anchor_idx, ind_list)
                    gt_mask[scale_id][prior_id][row_id, col_id, 5] = 1

                target_anchor_idx = np.argmax(iou_mat, axis=-1)[0]
                # target_anchor_batch[b_id].append(candidate_anchor[target_anchor_idx])
                anchor_idx = anchor_idx_list[target_anchor_idx]
                scale_id, prior_id, row_id, col_id, anchor_xyxy, anchor_stride =\
                    self._get_anchor_property(anchor_idx, ind_list)

                # gt_mask[scale_id][prior_id][row_id, col_id, :4] = gt_bbox
                gt_mask[scale_id][prior_id][row_id, col_id, :4] =\
                    bboxtool.xyxy2yolotcoord([gt_bbox], anchor_xyxy, anchor_stride, [col_id, row_id])
                gt_mask[scale_id][prior_id][row_id, col_id, 4] = 1
                gt_mask[scale_id][prior_id][row_id, col_id, 5] = 0
                # TODO
                # multi-class
                gt_mask[scale_id][prior_id][row_id, col_id, 6:] = one_hot_label[gt_id]
                # print(one_hot_label[gt_id])
            batch_gt_mask.append(gt_mask)

        if is_flatten:
            batch_gt_mask = self._flatten_gt_mask(batch_gt_mask)

        return np.array(batch_gt_mask)#, target_anchor_batch

    def _flatten_gt_mask(self, gt_mask_batch):
        flatten_gt_mask_batch = []
        for gt_mask in gt_mask_batch:
            flatten_gt_mask = []
            for scale_mask in gt_mask:
                for anchor_mask in scale_mask:
                    flatten_gt_mask.append(anchor_mask.reshape(-1, self._yolo_single_out_dim))
                    # print(anchor_mask.reshape(-1, self._yolo_single_out_dim).shape)
            flatten_gt_mask = np.concatenate(flatten_gt_mask, axis=-2)
            flatten_gt_mask_batch.append(flatten_gt_mask)
        return flatten_gt_mask_batch

    def _convert_gt(self, gt_bbox, im_shape, rescale_shape):
        gt_bbox_para = np.array([bbox[1:] for bbox in gt_bbox])
        gt_bbox_label = [int(bbox[0]) for bbox in gt_bbox]
        gt_bbox_para = bboxtool.rescale_bbox(gt_bbox_para, im_shape, rescale_shape)
        return gt_bbox_para, gt_bbox_label            

    def _get_anchor_property(self, anchor_idx, ind_list):
        index_dict = ind_list[anchor_idx]
        scale_id = index_dict['scale_id']
        row_id = index_dict['row_id']
        col_id = index_dict['col_id']
        prior_id = index_dict['prior_id']
        anchor_xyxy = index_dict['anchor']
        anchor_stride = index_dict['stride']
        return scale_id, prior_id, row_id, col_id, anchor_xyxy, anchor_stride

def get_target_anchor(gt_bbox, im_shape, rescale_shape, stride_list, prior_list, n_class):
    # gt_bbox xyxy
    # stride_list [n_scale]
    # anchor_list [n_scale, n_anchor]

    if isinstance(rescale_shape, int):
        rescale_shape = [rescale_shape, rescale_shape]
    if isinstance(im_shape, int):
        im_shape = [im_shape, im_shape]

    gt_bbox_para = np.array([bbox[1:] for bbox in gt_bbox])
    gt_bbox_label = [int(bbox[0]) for bbox in gt_bbox]
    # rescale gt_bbox
    # print(gt_bbox_para,im_shape, rescale_shape)
    gt_bbox_para = bboxtool.rescale_bbox(gt_bbox_para, im_shape, rescale_shape)

    # gt_mask = [np.zeros(rescale_shape[0] / stride, rescale_shape[0] / stride) for _ in range()
    #            for stride in stride_list]
    gt_mask = [[] for _ in range(len(stride_list))]
    for scale_id, (stride, anchor) in enumerate(zip(stride_list, prior_list)):
        for prior_id in range(len(anchor)):
            gt_mask[scale_id].append(np.zeros((int(rescale_shape[0] / stride),
                                               int(rescale_shape[0] / stride),
                                               1 + 4 + n_class)))
    ind_list = []
    anchor_list = []
    for scale_id, (stride, anchor) in enumerate(zip(stride_list, prior_list)):
        cols, rows = int(rescale_shape[0] / stride), int(rescale_shape[1] / stride)
        for row_id in range(rows):
            for col_id in range(cols):
                for prior_id, cur_anchor in enumerate(anchor):
                    
                    cur_index = {'stride': stride, 'prior': cur_anchor,
                                 'row_id': row_id, 'col_id': col_id,
                                 'scale_id': scale_id, 'prior_id': prior_id}
                    # cur_index = (stride, cur_anchor, row_id, col_id)
                    offset_anchor = [stride * col_id - cur_anchor[0] / 2,
                                     stride * row_id - cur_anchor[1] / 2,
                                     stride * col_id + cur_anchor[0] / 2,
                                     stride * row_id + cur_anchor[1] / 2]
                    ind_list.append(cur_index)
                    anchor_list.append(offset_anchor)

    iou_mat = bboxtool.bbox_list_IOU(gt_bbox_para, anchor_list, align=False)
    gt_anchor_idx = np.argmax(iou_mat, axis=-1)

    gt_anchor = np.array([anchor_list[i] for i in gt_anchor_idx])
    gt_label = np.array(gt_bbox_label)

    gt_dict = {}
    one_hot_label = vec2onehot(gt_bbox_label, n_class)
    for idx, anchor_idx in enumerate(gt_anchor_idx):
        anchor_dict = {}
        anchor_dict['anchor_para'] = anchor_list[anchor_idx]
        anchor_dict['gt_bbox_para'] = list(gt_bbox_para[idx])
        anchor_dict['label'] = gt_bbox_label[idx]
        index_dict = ind_list[anchor_idx]
        gt_dict[idx] = {**anchor_dict, **index_dict}

        scale_id = index_dict['scale_id']
        row_id = index_dict['row_id']
        col_id = index_dict['col_id']
        prior_id = index_dict['prior_id']

        gt_mask[scale_id][prior_id][row_id, col_id, :4] = gt_bbox_para[idx]
        gt_mask[scale_id][prior_id][row_id, col_id, 4] = 1
        gt_mask[scale_id][prior_id][row_id, col_id, 5:] = one_hot_label[idx]

        # print(gt_mask[scale_id][prior_id][row_id, col_id, :4])

        # print([np.array(gt_mask[i]).shape for i in range(3)])

    return gt_anchor, gt_label, gt_dict, np.array(gt_mask)
