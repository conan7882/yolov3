#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: preprocess.py
# Author: Qian Ge <geqian1001@gmail.com>

import copy
import numpy as np
import tensorflow as tf
import src.bbox.bboxtool as bboxtool
from src.utils.dataflow import vec2onehot
from src.dataflow.base import DetectionDataFlow
import src.dataflow.augmentation as augment
import src.utils.image as imagetool


class PreProcess(object):
    def __init__(self, dataflow, rescale_shape_list, stride_list, prior_list, n_class,
                 h_flip=False, crop=False, color=False, affine=False, im_intensity = 1.,
                 max_num_bbox_per_im=45):

        def _augment(im, bbox):

            im = np.array(im)
            im_h, im_w = im.shape[0], im.shape[1]
            if h_flip and np.random.random() > 0.5:
                im, bbox = augment.horizontal_flip(im, bbox)
            if crop:
                new_h = int(im_h * np.random.uniform(low=0.7, high=1.))
                new_w = int(im_w * np.random.uniform(low=0.7, high=1.))
                if np.random.random() > 0.5:
                    im, bbox = augment.center_crop(im, bbox, [new_h, new_w])
                else:
                    start_y = int(np.random.uniform(low=0., high=im_h - new_h))
                    start_x = int(np.random.uniform(low=0., high=im_w - new_w))
                    im, bbox = augment.crop(im, bbox, [start_y, start_x, new_h, new_w])
            if color:
                hue = np.random.uniform(low=-0.2, high=0.2)
                saturate = np.random.uniform(low=0.9, high=1.1)
                brightness = np.random.uniform(low=0.5, high=1)
                im = augment.change_color(im, hue, saturate, brightness, intensity_scale=im_intensity)
            if affine:
                a_scale = np.random.uniform(low=0.9, high=1.1, size=2)
                # scale_y = np.random.uniform(low=0.9, high=1.1, size=2)
                a_trans = np.random.uniform(low=-0.3, high=0.3, size=2)
                # t_y = np.random.uniform(low=-0.3, high=0.3)
                a_shear = np.random.uniform(low=-0.2, high=0.2, size=2)
                # s_y = np.random.uniform(low=-0.2, high=0.2)
                angle = np.random.uniform(low=-15, high=15)
                im, bbox = augment.affine_transform(
                    im, bbox, scale=a_scale, translation=a_trans, shear=a_shear, angle=angle)

            im = augment.im_preserve_range(im, im_intensity)
            bbox = augment.remove_invalid_bbox(im, bbox)

            return im, bbox, (im_h, im_w)

        self._aug_fnc = _augment
        self._max_bbox = max_num_bbox_per_im
        self._h_flip = h_flip
        self._crop = crop
        self._color = color 
        self._affine = affine
        self._im_intensity = im_intensity

        assert isinstance(dataflow, DetectionDataFlow)
        self.dataflow = dataflow
        anchor_list = np.reshape(prior_list, (-1,2))
        # cxywh
        self._anchor_boxes = np.concatenate(
            (np.zeros((len(anchor_list), 2)), anchor_list), axis=-1)

        self._n_class = n_class
        self._yolo_single_out_dim = 1 + 4 + n_class # 1: obj 4: bbox
        self._stride_list = stride_list

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
                            ind_list.append(cur_index)

                            sub2ind_dict.setdefault((scale_id, row_id, col_id), []).append(anchor_cnt)
                            anchor_cnt += 1
            self.init_anchors_dict[rescale_shape[0]] = {'index': ind_list, 'sub2ind': sub2ind_dict}

    def set_output_scale(self, scale):
        if isinstance(scale, int):
            self._output_scale = [scale, scale]
        else:
            assert len(scale) == 2
            self._output_scale = scale

    def tf_process_batch(self, batch_im, batch_labels):
        output_scale = self._output_scale

        def _process_batch(batch_im, batch_labels):
            true_boxes = np.zeros([len(batch_im), self._max_bbox, 4])
            true_classes = -1. * np.ones([len(batch_im), self._max_bbox])
            gt_mask_batch = []
            im_batch = []
            for idx, (im, labels) in enumerate(zip(batch_im, batch_labels)):
                bboxes = np.array([bbox[1:] for bbox in labels])
                class_labels = [int(bbox[0]) for bbox in labels]
                im, bboxes, im_shape = self._aug_fnc(im, bboxes)

                im, bboxes = augment.rescale(im, bboxes, output_scale)

                im_batch.append(im)
                true_boxes[idx, :len(bboxes)] = bboxes
                true_classes[idx, :len(class_labels)] = class_labels
                
                gt_mask = self._get_gt_mask(bboxes, class_labels, output_scale)
                gt_mask_batch.append(gt_mask)

            gt_mask_batch = self._flatten_gt_mask(gt_mask_batch)
            return np.array(im_batch).astype(np.float32),\
                   np.array(gt_mask_batch).astype(np.float32),\
                   true_boxes.astype(np.float32),\
                   true_classes.astype(np.float32)

        im_b, gt_mask_b, boxes_b , classes_b = tf.py_func(
            _process_batch,
            [batch_im, batch_labels],
            [tf.float32, tf.float32, tf.float32, tf.float32],
            name="map_fnc")

        im = tf.reshape(im_b[0], (output_scale[0], output_scale[1], batch_im.shape[-1]))
        gt_mask = tf.reshape(gt_mask_b[0], (-1, self._yolo_single_out_dim))
        bboxes = tf.reshape(boxes_b[0], (self._max_bbox, 4))
        classes = tf.reshape(classes_b[0], (self._max_bbox,))
        return im, gt_mask, bboxes, classes

    def process_batch(self, output_scale):
        batch_data = self.dataflow.next_batch_dict()

        true_boxes = np.zeros([len(batch_data['image']), self._max_bbox, 4])
        gt_mask_batch = []
        im_batch = []
        for idx, (im, labels) in enumerate(zip(batch_data['image'], batch_data['label'])):
            bboxes = np.array([bbox[1:] for bbox in labels])
            class_labels = [int(bbox[0]) for bbox in labels]
            im, bboxes, im_shape = self._aug_fnc(im, bboxes)

            im, bboxes = augment.rescale(im, bboxes, output_scale)

            im_batch.append(im)
            true_boxes[idx, :len(bboxes)] = bboxes
            
            gt_mask = self._get_gt_mask(bboxes, class_labels, output_scale)
            gt_mask_batch.append(gt_mask)

        gt_mask_batch = self._flatten_gt_mask(gt_mask_batch)
        return np.array(im_batch), np.array(gt_mask_batch), true_boxes

    def _get_gt_mask(self, gt_bboxes, class_labels, rescale_shape):

        init_anchors = self.init_anchors_dict[rescale_shape[0]]
        ind_list = init_anchors['index']
        sub2ind = init_anchors['sub2ind']

        gt_mask = copy.deepcopy(self.init_gt_mask_dict[rescale_shape[0]])
        one_hot_label = vec2onehot(class_labels, self._n_class)
        gt_cxy = np.stack(
            [(gt_bboxes[:, 0] + gt_bboxes[:, 2]) // 2,
             (gt_bboxes[:, 1] + gt_bboxes[:, 3]) // 2], axis=-1)

        iou_mat = bboxtool.bbox_list_IOU(
            gt_bboxes, bboxtool.cxywh2xyxy(self._anchor_boxes), align=True)
        target_anchor_list = np.argmax(iou_mat, axis=-1)
        # print()
        # out_anchor_list = []
        for gt_id, (target_anchor_idx, gt_bbox) in enumerate(zip(target_anchor_list, gt_bboxes)):
            if iou_mat[gt_id, target_anchor_idx] == 0:
                continue
            anchor_idx_list = []
            for scale_id, stride in enumerate(self._stride_list):
                anchor_feat_cxy = gt_cxy[gt_id] // stride
                gt_feat_cxy =  gt_cxy[gt_id] / stride

                # print(gt_bboxes[gt_id],gt_cxy[gt_id])
                anchor_idx_list += sub2ind[(scale_id, anchor_feat_cxy[1], anchor_feat_cxy[0])]

            anchor_idx = anchor_idx_list[target_anchor_idx]
            scale_id, prior_id, row_id, col_id, anchor_xyxy, anchor_stride =\
                self._get_anchor_property(anchor_idx, ind_list)

            gt_mask[scale_id][prior_id][row_id, col_id, :4] =\
                bboxtool.xyxy2yolotcoord([gt_bbox], anchor_xyxy, anchor_stride, [col_id, row_id])
            gt_mask[scale_id][prior_id][row_id, col_id, 4] = 1
            # TODO
            # multi-class
            gt_mask[scale_id][prior_id][row_id, col_id, 5:] = one_hot_label[gt_id]
            # out_anchor_list.append(anchor_list[anchor_idx])
            # out_anchor_list.append([col_id*anchor_stride, row_id*anchor_stride] + anchor_xyxy)
            # print(anchor_list[anchor_idx])
            # print(col_id*anchor_stride, row_id*anchor_stride)

        return gt_mask#, out_anchor_list

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

    def _convert_label(self, labels, im_shape, rescale_shape):
        gt_bbox_para = np.array([bbox[1:] for bbox in labels])
        gt_bbox_label = [int(bbox[0]) for bbox in labels]
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

    