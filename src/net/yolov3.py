#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: yolov3.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np
import tensorflow as tf

import src.utils.viz as viz
import src.model.layers as L
import src.model.losses as losses
from src.net.base import BaseModel
# from src.bbox.bboxgt import TargetAnchor
import src.utils.image as imagetool
import src.model.yolo_module as yolo_module
import src.model.darknet_module as darknet_module
import src.model.detection_bbox as detection_bbox

INIT_W = tf.keras.initializers.he_normal()

class YOLOv3(BaseModel):
    def __init__(self,
                 n_channel,
                 n_class, 
                 pre_trained_path, 
                 anchors,
                 bsize=2,
                 obj_score_thr=0.8,
                 nms_iou_thr=0.45,
                 feature_extractor_trainable=False,
                 detector_trainable=False):

        self._n_channel = n_channel
        self._n_class = n_class
        self._feat_trainable = feature_extractor_trainable
        self._dete_trainable = detector_trainable
        self._bsize = bsize
        # self._rescale_shape = L.get_shape2D(rescale_shape)

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
        self.o_shape = tf.placeholder(tf.float32, [None, 2], 'o_shape')
        self.image = tf.placeholder(
            tf.float32, [None, None, None, self._n_channel], name='image')
        self.label = tf.placeholder(
            tf.float32, [None, None, (1 + 4 + 1 + self._n_class)], 'label')
        self.lr = tf.placeholder(tf.float32, name='lr')
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
        _, _, self.layers['bbox_t_coord'], self.layers['objectness_logits'], self.layers['classes_logits']\
            = self._create_model(self.image)
        self.loss = self._get_loss()

        self.train_op = self.get_train_op()
        self.loss_op = self.get_loss()
        self.global_step = 0
        self.epoch_id = 0

        # self.layers['gt_mask'] = self._get_target_anchor(self._rescale_shape)

    def create_valid_model(self):
        self.set_is_training(is_training=False)
        self._create_valid_input()
        self.layers['bbox_score'], self.layers['bbox'], _, _, _\
            = self._create_model(self.image)

        self.epoch_id = 0

        # self.layers['det_score'], self.layers['det_bbox'], self.layers['det_class'] =\
        #     self._get_detection(self.layers['bbox_score'], self.layers['bbox'])

    def _create_model(self, inputs):
        with tf.variable_scope('DarkNet53', reuse=tf.AUTO_REUSE):
            feat_out, route_1, route_2 = darknet_module.darkent53_conv(
                inputs, pretrained_dict=self._pretrained_dict,
                init_w=INIT_W, bn=True, wd=0, trainable=self._feat_trainable,
                is_training=self.is_training, name='darkent53_conv')
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

            def make_tensor_and_batch_flatten(inputs, last_dim):
                inputs = tf.concat(inputs, axis=1) # [bsize, -1, last_dim] 
                return tf.reshape(inputs, (bsize, -1, last_dim)) # [bsize, -1, n_class]

            bbox_score_list = make_tensor_and_batch_flatten(bbox_score_list, self._n_class)
            classes_pred_list = make_tensor_and_batch_flatten(classes_pred_list, self._n_class)
            bbox_list = make_tensor_and_batch_flatten(bbox_list, 4)
            bbox_t_coord_list = make_tensor_and_batch_flatten(bbox_t_coord_list, 4)
            objectness_list = make_tensor_and_batch_flatten(objectness_list, 1)

            return bbox_score_list, bbox_list, bbox_t_coord_list, objectness_list, classes_pred_list

    def _get_loss(self):
        with tf.name_scope('loss'):
            bsize = tf.cast(tf.shape(self.o_shape)[0], tf.float32)

            bbox_label, objectness_label, ignore_mask, classes_label = tf.split(
                self.label, [4, 1, 1, self._n_class], axis=-1, name='split_label')

            cls_loss = losses.class_loss(
                classes_label, self.layers['classes_logits'], objectness_label)
            bbox_loss = losses.bboxes_loss(
                bbox_label, self.layers['bbox_t_coord'], objectness_label)

            obj_loss = losses.objectness_loss(
                objectness_label, self.layers['objectness_logits'], ignore_mask)

            # print(cls_loss, bbox_loss, obj_loss)
            loss = (10 * cls_loss + 10 * bbox_loss + obj_loss) / bsize
            self.cls_loss = cls_loss / bsize
            self.bbox_loss = bbox_loss / bsize
            self.obj_loss = obj_loss / bsize
        return loss

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.lr)

    def train_epoch(self, sess, train_data, target_anchor, init_lr, im_rescale,
                    summary_writer=None):
        display_name_list = ['cls_loss', 'bbox_loss', 'obj_loss', 'loss']
        cur_summary = None
        lr = init_lr

        cls_loss_sum = 0
        bbox_loss_sum = 0
        obj_loss_sum = 0
        loss_sum = 0
        step = 0

        self.epoch_id += 1
        cur_epoch = train_data.epochs_completed
        while train_data.epochs_completed == cur_epoch:
            step += 1
            self.global_step += 1

            batch_data = train_data.next_batch_dict()
            batch_gt = target_anchor.get_yolo_target_anchor(
                batch_data['label'], batch_data['shape'], im_rescale, True)

            _, loss, cls_loss, bbox_loss, obj_loss = sess.run(
                [self.train_op, self.loss_op, self.cls_loss,
                 self.bbox_loss, self.obj_loss],
                feed_dict={self.image: batch_data['image'],
                           self.o_shape: batch_data['shape'],
                           self.label: batch_gt,
                           self.lr: lr})

            cls_loss_sum += cls_loss
            bbox_loss_sum += bbox_loss
            obj_loss_sum += obj_loss
            loss_sum += loss
            
            if step % 10 == 0:
                viz.display(
                    self.global_step,
                    step,
                    [cls_loss_sum, bbox_loss_sum, obj_loss_sum, loss_sum],
                    display_name_list,
                    'train',
                    summary_val=cur_summary,
                    summary_writer=summary_writer)

        print('==== epoch: {}, lr:{} ===='.format(cur_epoch, lr))
        viz.display(
            self.global_step,
            step,
            [cls_loss_sum, bbox_loss_sum, obj_loss_sum, loss_sum],
            display_name_list,
            'train',
            summary_val=cur_summary,
            summary_writer=summary_writer)

    def predict_epoch_or_step(self, sess, dataflow, im_rescale,
                              obj_score_thr, nms_iou_thr,
                              label_dict, category_index, save_path,
                              run_type='step'):

        im_id = 0
        def run_step():
            nonlocal im_id
            batch_data = dataflow.next_batch_dict()
            bbox_score, bbox_para = sess.run(
                [self.layers['bbox_score'], self.layers['bbox']],
                feed_dict={self.image: batch_data['image']})
            for idx, (score, para) in enumerate(zip(bbox_score, bbox_para)):
                
                nms_boxes, nms_scores, nms_label_names, nms_label_ids =\
                    detection_bbox.detection(
                        para, score, n_class=self._n_class, obj_score_thr=obj_score_thr,
                        nms_iou_thr=nms_iou_thr, label_dict=label_dict,
                        image_shape=batch_data['shape'][idx], rescale_shape=im_rescale)
                original_im = imagetool.rescale_image(
                    batch_data['image'][idx] * 255, batch_data['shape'][idx])
                viz.draw_bounding_box_on_image_array(
                    original_im, bbox_list=nms_boxes, class_list=nms_label_ids,
                    score_list=nms_scores, category_index=category_index,
                    save_name=os.path.join(save_path, 'epoch_{}_im_{}.png'.format(self.epoch_id, im_id)),
                    save_fig=True)
                im_id += 1
                print(nms_label_names)
        im_id = 0
        if run_type == 'step':
            run_step()
        elif run_type == 'epoch':
            dataflow.reset_epochs_completed()
            while dataflow.epochs_completed == 0:
                run_step()
            self.epoch_id += 1
        else:
            raise ValueError("Invalid run_type: {}! Has to be 'step' or 'epoch'.".format(run_type))


    # def _get_detection(self, bbox_score_list, bbox_para_list):
    #     det_score_list = []
    #     det_bbox_list = []
    #     det_class_list = []

    #     for batch_id in range(self._bsize):
    #         bbox_score = bbox_score_list[batch_id]
    #         bbox_para = bbox_para_list[batch_id]

    #         det_score, det_bbox, det_class = yolo_module.get_detection(
    #             bbox_score, bbox_para, self._n_class,
    #             self._obj_score_thr, self._nms_iou_thr, max_boxes=20,
    #             rescale_shape=self._rescale_shape, original_shape=self.o_shape[batch_id])
    #         det_score_list.append(det_score)
    #         det_bbox_list.append(det_bbox)
    #         det_class_list.append(det_class)
    #     return det_score_list, det_bbox_list, det_class_list

