#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: yolov3.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from src.net.base import BaseModel
import src.model.layers as L
import src.model.darknet_module as darknet_module
import src.model.yolo_module as yolo_module
# import src.utils.viz as viz
from src.bbox.bboxgt import TargetAnchor
import src.model.losses as losses

INIT_W = None

class YOLOv3(BaseModel):
    def __init__(self, n_channel, n_class, pre_trained_path, anchors,
                 bsize=2, obj_score_thr=0.8, nms_iou_thr=0.45,
                 feature_extractor_trainable=False, detector_trainable=False,
                 rescale_shape=None):
        self._n_channel = n_channel
        self._n_class = n_class
        self._feat_trainable = feature_extractor_trainable
        self._dete_trainable = detector_trainable
        self._bsize = bsize
        self._rescale_shape = L.get_shape2D(rescale_shape)

        self._obj_score_thr = obj_score_thr
        self._nms_iou_thr = nms_iou_thr

        self._anchors = anchors
        self._stride_list = [32, 16, 8]

        # self._n_scale = len(self._stride_list)
        # self._n_anchor = len(self._anchors[0])
        # self._n_yolo_out_dim = (1 + 4 + self._n_class) * self._n_anchor

        # if detector_trainable:
        #     self.target_anchor = TargetAnchor(
        #         [self._rescale_shape[0]], self._stride_list, self._anchors, self._n_class)

        self._pretrained_dict = None
        if pre_trained_path:
            self._pretrained_dict = np.load(
                pre_trained_path, encoding='latin1').item()

        self.layers = {}

    def _create_train_input(self):
        self.image = tf.placeholder(
            tf.float32, [None, None, None, self._n_channel], name='image')
        self.o_shape = tf.placeholder(tf.float32, [None, 2], 'o_shape')
        self.label = tf.placeholder(
            tf.float32,
            [None, None, (1 + 4 + 1 + self._n_class)],
            'label')
        # self.label = tf.placeholder(tf.int64, [None, 2], 'label')
        self.keep_prob = 1.

    def _create_valid_input(self):
        self.image = tf.placeholder(
            tf.float32, [None, None, None, self._n_channel], name='image')
        self.o_shape = tf.placeholder(tf.float32, [None, 2], 'o_shape')
        # self.label = tf.placeholder(tf.int64, [None, 2], 'label')
        self.keep_prob = 1.

    def create_train_model(self):
        self.set_is_training(is_training=True)
        self._create_train_input()
        self.layers['bbox_score'], self.layers['bbox'], self.layers['bbox_t_coord'], self.layers['objectness_logits'], self.layers['classes_logits']\
            = self._create_model(self.image)
        self.loss = self._get_loss()
        # self.layers['gt_mask'] = self._get_target_anchor(self._rescale_shape)

    def create_valid_model(self):
        self.set_is_training(is_training=False)
        self._create_valid_input()
        self.layers['bbox_score'], self.layers['bbox'], _, _, _\
            = self._create_model(self.image)

        # self.layers['det_score'], self.layers['det_bbox'], self.layers['det_class'] =\
        #     self._get_detection(self.layers['bbox_score'], self.layers['bbox'])

    def _create_model(self, inputs):
        with tf.variable_scope('DarkNet53', reuse=tf.AUTO_REUSE):
            feat_out, route_1, route_2 = darknet_module.darkent53_conv(
                inputs, pretrained_dict=self._pretrained_dict, init_w=INIT_W,
                bn=True, wd=0, trainable=self._feat_trainable, is_training=self.is_training,
                name='darkent53_conv')
            darknetFeat_list = [None, route_1, route_2]

        with tf.variable_scope('yolo_prediction', reuse=tf.AUTO_REUSE):
            
            out_dim_list = [1024, 512, 256]
            bbox_score_list = []
            bbox_list = []
            bbox_t_coord_list = []
            objectness_list = []
            classes_pred_list = []
            prev_feat = feat_out
            for scale_id, anchors in enumerate(self._anchors):
                if scale_id > 0:
                    up_sample = True
                else:
                    up_sample = False

                n_anchor = len(anchors)
                linear_out_dim = (1 + 4 + self._n_class) * n_anchor

                scale = self._stride_list[scale_id]
                yolo_out, prev_feat = yolo_module.yolo_layer(
                    prev_feat, darknet_feat=darknetFeat_list[scale_id], up_sample=up_sample,
                    n_filter=out_dim_list[scale_id], out_dim=linear_out_dim,
                    pretrained_dict=self._pretrained_dict, init_w=INIT_W,
                    bn=True, wd=0, trainable=self._dete_trainable, is_training=self.is_training,
                    scale_id=scale_id+1, name='yolo')
                bbox_score, bbox, bbox_t_coord, objectness_logits, classes_pred_logits = yolo_module.yolo_prediction(
                    yolo_out, anchors, self._n_class, scale, scale_id+1, name='yolo_prediction')
                
                bbox_score_list.append(bbox_score)
                bbox_list.append(bbox)
                objectness_list.append(objectness_logits)
                classes_pred_list.append(classes_pred_logits)
                bbox_t_coord_list.append(bbox_t_coord)

            #bbox_score_list [n_scale, bsize, -1, n_class] 
            bsize = tf.shape(inputs)[0]

            bbox_score_list = tf.concat(bbox_score_list, axis=1) # [bsize, -1, n_class] 
            bbox_score_list = tf.reshape(bbox_score_list, (bsize, -1, self._n_class)) # [bsize, -1, n_class]

            bbox_list = tf.concat(bbox_list, axis=1)
            bbox_list = tf.reshape(bbox_list, (bsize, -1, 4))

            bbox_t_coord_list = tf.concat(bbox_t_coord_list, axis=1)
            bbox_t_coord_list = tf.reshape(bbox_t_coord_list, (bsize, -1, 4))

            classes_pred_list = tf.concat(classes_pred_list, axis=1) # [bsize, -1, n_class] 
            classes_pred_list = tf.reshape(classes_pred_list, (bsize, -1, self._n_class)) # [bsize, -1, n_class]

            objectness_list = tf.concat(objectness_list, axis=1)
            objectness_list = tf.reshape(objectness_list, (bsize, -1, 1))

            return bbox_score_list, bbox_list, bbox_t_coord_list, objectness_list, classes_pred_list

    def _get_detection(self, bbox_score_list, bbox_para_list):
        det_score_list = []
        det_bbox_list = []
        det_class_list = []

        for batch_id in range(self._bsize):
            bbox_score = bbox_score_list[batch_id]
            bbox_para = bbox_para_list[batch_id]

            det_score, det_bbox, det_class = yolo_module.get_detection(
                bbox_score, bbox_para, self._n_class,
                self._obj_score_thr, self._nms_iou_thr, max_boxes=20,
                rescale_shape=self._rescale_shape, original_shape=self.o_shape[batch_id])
            det_score_list.append(det_score)
            det_bbox_list.append(det_bbox)
            det_class_list.append(det_class)
        return det_score_list, det_bbox_list, det_class_list

    def _get_loss(self):
        with tf.name_scope('loss'):
            bbox_label, objectness_label, ignore_mask, classes_label = tf.split(
                self.label, [4, 1, 1, self._n_class], axis=-1, name='split_label')

            cls_loss = losses.class_loss(
                classes_label, self.layers['classes_logits'], objectness_label)
            bbox_loss = losses.bboxes_loss(
                bbox_label, self.layers['bbox_t_coord'], objectness_label)

            obj_loss = losses.objectness_loss(
                objectness_label, self.layers['objectness_logits'], ignore_mask)

            loss = cls_loss
        return loss

    # def _get_target_anchor(self, rescale):
    #     # self.target_anchor.get_target_anchor(
    #     #     gt_bbox_batch, im_shape_batch, rescale)

    #     gt_mask =tf.py_func(
    #         self.target_anchor.get_target_anchor,
    #         [self.label, self.o_shape, rescale,],
    #         tf.float32,
    #         name="target_anchors")

    #     return gt_mask


