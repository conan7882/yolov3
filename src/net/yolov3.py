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
import src.utils.image as imagetool
import src.model.yolo_module as yolo_module
import src.model.darknet_module as darknet_module
import src.model.detection_bbox as detection_bbox
import src.bbox.tfbboxtool as tfbboxtool


INIT_W = tf.keras.initializers.he_normal()

class YOLOv3(BaseModel):
    """ model for yolov3 inference and training """
    def __init__(self,
                 n_channel,
                 n_class, 
                 pre_trained_path, 
                 anchors,
                 bsize=2,
                 ignore_thr=0.5,
                 feature_extractor_trainable=False,
                 detector_trainable=False,
                 obj_weight=1.,
                 nobj_weight=1.,
                 category_index=None):
        """ 
            Args:
                n_channel (int): number of channels of input image
                n_class (int): number of classes
                pre_trained_path (str): path of pretrained model (.npy file)
                anchors (List[List[float]]): clustering anchor box
                bsize (int): batch size
                obj_score_thr (float): confidence score threshold for selecting detected bbox
                nms_iou_thr (float): IoU threshold for non maximum suppression
                feature_extractor_trainable (bool): whether train feature extractor or not
                detector_trainable (bool): whether train yolo layer or not
        """

        self._n_channel = n_channel
        self._n_class = n_class
        self._feat_trainable = feature_extractor_trainable
        self._dete_trainable = detector_trainable
        self._bsize = bsize

        self._anchors = anchors
        self._stride_list = [32, 16, 8]
        self._ignore_thr = ignore_thr

        # adjust obj and nobj scale
        self._obj_w = obj_weight
        self._pos_obj_w = obj_weight * 1 / nobj_weight
        self._adjust_obj_w = nobj_weight

        # load pre-trained model
        self._pretrained_dict = None
        if pre_trained_path:
            self._pretrained_dict = np.load(
                pre_trained_path, encoding='latin1').item()

        if category_index is None:
            category_index = {}
            for class_id in range(n_class):

                category_index[class_id] = {'id': class_id, 'name': 'unk'}
        self._category_index = category_index
        self.layers = {}

    def _create_train_input(self, input_batch):
        """ receive and create training data

            Args:
                input_batch (tensor): Return of tf.data.Iterator.get_next() with length 3.
                    The order of data should be: image, label (gt_mask), true_boxes, true_classes
        """
        # input_batch: image, label (gt_mask), true_boxes, true_classes
        self.image, self.label, self.true_boxes, self.true_classes = input_batch
        self.lr = tf.placeholder(tf.float32, name='lr')

    def create_train_model(self, input_batch):
        """ create model for training

            Args:
                input_batch (tensor): Return of tf.data.Iterator.get_next() with length 3.
                    The order of data should be: image, label (gt_mask), true_boxes
        """
        self.set_is_training(is_training=True)
        self._create_train_input(input_batch)
        self.layers['pred_score'], self.layers['pred_bbox'], self.layers['bbox_t_coord'], self.layers['objectness_logits'], self.layers['classes_logits']\
            = self._create_model(self.image)

        self.layers['pred_im'], self.layers['true_im'] = self._get_bbox_on_image_tensor()

        self.train_op = self.get_train_op()
        self.loss_op = self.get_loss()
        self.train_summary_op = self.get_summary('train_images')
        self.global_step = 0
        self.epoch_id = 0

    def _get_bbox_on_image_tensor(self):
        pred_im = viz.tf_draw_bounding_box(
            im = self.image * 255., 
            bbox_list=self.layers['pred_bbox'], 
            score_list=tf.reduce_max(self.layers['pred_score'], axis=-1), 
            class_list=tf.argmax(self.layers['pred_score'], axis=-1), 
            category_index=self._category_index,
            max_boxes_to_draw=20, 
            min_score_thresh=0.2)

        true_im = viz.tf_draw_bounding_box(
            im = self.image * 255., 
            bbox_list=self.true_boxes, 
            score_list=tf.cast(self.true_classes > 0, tf.int64),
            class_list=self.true_classes * tf.cast(self.true_classes > 0, tf.float32), 
            category_index=self._category_index,
            max_boxes_to_draw=20, 
            min_score_thresh=0.5)

        return pred_im, true_im

    def get_summary(self, name):
        with tf.name_scope(name):
            tf.summary.image(
                'prediction',
                self.layers['pred_im'],
                collections=[name])
            tf.summary.image(
                'groundtruth',
                self.layers['true_im'],
                collections=[name])
            return tf.summary.merge_all(key=name)

    def _create_valid_input(self, input_batch):
        """ receive and create validation data

            Args:
                input_batch (tensor): Return of tf.data.Iterator.get_next() with length 3.
                    The order of data should be: image, label (gt_mask), true_boxes, true_classes
        """
        # input_batch: image, label (gt_mask), true_boxes, true_classes
        self.image, self.label, self.true_boxes, self.true_classes = input_batch

    def create_valid_model(self, input_batch):
        """ create model for validation

            Args:
                input_batch (tensor): Return of tf.data.Iterator.get_next() with length 3.
                    The order of data should be: image, label (gt_mask), true_boxes
        """
        self.set_is_training(is_training=True)
        self._create_valid_input(input_batch)
        self.layers['pred_score'], self.layers['pred_bbox'], self.layers['bbox_t_coord'], self.layers['objectness_logits'], self.layers['classes_logits']\
            = self._create_model(self.image)

        self.layers['pred_im'], self.layers['true_im'] = self._get_bbox_on_image_tensor()

        self.loss_op = self.get_loss()
        self.valid_summary_op = self.get_summary('valid_images')
        self.global_step = 0
        self.epoch_id = 0

    def _create_test_input(self):
        self.image = tf.placeholder(
            tf.float32, [None, None, None, self._n_channel], name='image')
        self.o_shape = tf.placeholder(tf.float32, [None, 2], 'o_shape')

    def create_test_model(self):
        self.set_is_training(is_training=False)
        self._create_test_input()
        self.layers['bbox_score'], self.layers['bbox'], _, _, _\
            = self._create_model(self.image)

        self.epoch_id = 0
        # self.layers['det_score'], self.layers['det_bbox'], self.layers['det_class'] =\
        #     self._get_detection(self.layers['bbox_score'], self.layers['bbox'])

    def _create_model(self, inputs):
        """ create the entire net of yolov3 """

        with tf.variable_scope('DarkNet53', reuse=tf.AUTO_REUSE):
            feat_out, route_1, route_2 = darknet_module.darkent53_conv(
                inputs, 
                init_w=INIT_W, 
                bn=True, wd=0, 
                trainable=self._feat_trainable,
                is_training=self.is_training, 
                pretrained_dict=self._pretrained_dict,
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
                    prev_feat, 
                    darknet_feat=darknetFeat_list[scale_id], 
                    up_sample=up_sample,
                    n_filter=out_dim_list[scale_id], 
                    out_dim=linear_out_dim,
                    init_w=INIT_W,
                    bn=True, wd=0, 
                    trainable=self._dete_trainable, 
                    is_training=self.is_training,
                    scale_id=scale_id + 1, 
                    pretrained_dict=self._pretrained_dict, 
                    name='yolo')

                bbox_score, bbox, bbox_t_coord, objectness_logits, classes_pred_logits =\
                    yolo_module.yolo_prediction(yolo_out, 
                                                anchors, 
                                                self._n_class, 
                                                scale, 
                                                scale_id + 1, 
                                                name='yolo_prediction')
                bbox_list.append(bbox)
                bbox_score_list.append(bbox_score)
                objectness_list.append(objectness_logits)
                classes_pred_list.append(classes_pred_logits)
                bbox_t_coord_list.append(bbox_t_coord)

            #bbox_score_list [n_scale, bsize, -1, n_class] 
            bsize = tf.shape(inputs)[0]

            def make_tensor_and_batch_flatten(inputs, last_dim):
                """ make list of tensor to be one tensor and batch flatten 

                    Args:
                        inputs (list of tensor): length of list is n_scale and 
                            the shape of each tensor is [bsize, -1, last_dim]

                    Returns:
                        a tensor contains prediction of all scales with shape [bsize, -1, last_dim]

                """
                inputs = tf.concat(inputs, axis=1) # [bsize, -1, last_dim] 
                return tf.reshape(inputs, (bsize, -1, last_dim)) # [bsize, -1, last_dim]

            bbox_score_list = make_tensor_and_batch_flatten(bbox_score_list, self._n_class)
            classes_pred_list = make_tensor_and_batch_flatten(classes_pred_list, self._n_class)
            bbox_list = make_tensor_and_batch_flatten(bbox_list, 4)
            bbox_t_coord_list = make_tensor_and_batch_flatten(bbox_t_coord_list, 4)
            objectness_list = make_tensor_and_batch_flatten(objectness_list, 1)

            return bbox_score_list, bbox_list, bbox_t_coord_list, objectness_list, classes_pred_list

    def _get_ignore_mask(self, pred_bbox, true_bbox, objectness_label):
        """ compute ignore mask based on the prediction
            Do not compute loss for non-taget predicted bboxes with IoU greater than a thr.

            Args:
                pred_bbox (tensor): prediction bbox with shape [bsize, n_prediction, 4]
                true_bbox (tensor): ground truth bbox with shape [bsize, max_bbox_per_im, 4]
                objectness_label (tensor): label of target anchors with shape [bsize, n_prediction, 1]

            Returns:
                ignore mask which masks out non-target bbox with large enough IoU
                with shape [bsize, n_prediction, 1]

        """
        # [bdize, n_prediction, max_bbox_per_im]
        iou = tfbboxtool.batch_bbox_IoU(pred_bbox, true_bbox)
        best_ious = tf.reduce_max(iou, axis=-1, keepdims=True)
        ignore_mask = tf.logical_or(tf.cast(objectness_label, tf.bool),
                                    (best_ious < self._ignore_thr))
        return tf.to_float(ignore_mask)

    def _get_loss(self):
        with tf.name_scope('loss'):
            bsize = tf.cast(tf.shape(self.true_boxes)[0], tf.float32)

            bbox_label, objectness_label, classes_label = tf.split(
                self.label, [4, 1, self._n_class], axis=-1, name='split_label')

            ignore_mask = self._get_ignore_mask(self.layers['pred_bbox'],
                                                self.true_boxes,
                                                objectness_label)

            obj_loss = losses.objectness_loss(objectness_label,
                                              self.layers['objectness_logits'],
                                              ignore_mask,
                                              self._pos_obj_w)

            cls_loss = losses.class_loss(classes_label,
                                         self.layers['classes_logits'],
                                         objectness_label)

            bbox_loss = losses.bboxes_loss(bbox_label,
                                           self.layers['bbox_t_coord'],
                                           objectness_label)
            
            loss = (self._obj_w * cls_loss\
                    + self._obj_w * bbox_loss\
                    + self._adjust_obj_w * obj_loss) / bsize
            
            self.cls_loss = cls_loss / bsize
            self.bbox_loss = bbox_loss / bsize
            self.obj_loss = obj_loss / bsize
        return loss

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.lr)

    def train_epoch(self, sess, init_lr, summary_writer=None):
        """ Train the model for one epoch

            Args:
                sess (tf.Session())
                init_lr (float): learning rate
                summary_writer (tf.summary)
        """

        display_name_list = ['cls_loss', 'bbox_loss', 'obj_loss', 'loss']
        cur_summary = None
        lr = init_lr

        cls_loss_sum = 0
        bbox_loss_sum = 0
        obj_loss_sum = 0
        loss_sum = 0
        step = 0

        self.epoch_id += 1
        while True:
            try:
                step += 1
                self.global_step += 1
                
                if step % 100 == 0:
                    _, loss, cls_loss, bbox_loss, obj_loss, cur_summary = sess.run(
                        [self.train_op, self.loss_op, self.cls_loss,
                         self.bbox_loss, self.obj_loss, self.train_summary_op],
                        feed_dict={self.lr: lr})

                    viz.display(
                        self.global_step,
                        step,
                        [cls_loss_sum, bbox_loss_sum, obj_loss_sum, loss_sum],
                        display_name_list,
                        'train',
                        summary_val=cur_summary,
                        summary_writer=summary_writer)

                else:
                    _, loss, cls_loss, bbox_loss, obj_loss = sess.run(
                        [self.train_op, self.loss_op, self.cls_loss,
                         self.bbox_loss, self.obj_loss],
                        feed_dict={self.lr: lr})

                cls_loss_sum += cls_loss
                bbox_loss_sum += bbox_loss
                obj_loss_sum += obj_loss
                loss_sum += loss

            except tf.errors.OutOfRangeError:
                break

        # write summary 
        print('==== epoch: {}, lr:{} ===='.format(self.epoch_id, lr))
        viz.display(
            self.global_step,
            step,
            [cls_loss_sum, bbox_loss_sum, obj_loss_sum, loss_sum],
            display_name_list,
            'train',
            summary_val=None,
            summary_writer=summary_writer)

    def valid_epoch(self, sess, summary_writer=None):
        """ Train the model for one epoch

            Args:
                sess (tf.Session())
                init_lr (float): learning rate
                summary_writer (tf.summary)
        """

        display_name_list = ['cls_loss', 'bbox_loss', 'obj_loss', 'loss']
        cur_summary = None

        cls_loss_sum = 0
        bbox_loss_sum = 0
        obj_loss_sum = 0
        loss_sum = 0
        step = 0

        self.epoch_id += 1

        step += 1
        self.global_step += 1
        loss, cls_loss, bbox_loss, obj_loss, cur_summary = sess.run(
            [self.loss_op, self.cls_loss,
             self.bbox_loss, self.obj_loss,
             self.valid_summary_op])

        cls_loss_sum += cls_loss
        bbox_loss_sum += bbox_loss
        obj_loss_sum += obj_loss
        loss_sum += loss

        while True:
            try:
                step += 1
                self.global_step += 1

                loss, cls_loss, bbox_loss, obj_loss = sess.run(
                    [self.loss_op, self.cls_loss,
                     self.bbox_loss, self.obj_loss])

                cls_loss_sum += cls_loss
                bbox_loss_sum += bbox_loss
                obj_loss_sum += obj_loss
                loss_sum += loss
                
            except tf.errors.OutOfRangeError:
                break

        # write summary 
        print('[valid]:', end='')
        viz.display(
            self.epoch_id,
            step,
            [cls_loss_sum, bbox_loss_sum, obj_loss_sum, loss_sum],
            display_name_list,
            'valid',
            summary_val=cur_summary,
            summary_writer=summary_writer)

    def predict_epoch_or_step(self, 
                              sess, 
                              dataflow, 
                              im_rescale,
                              obj_score_thr, 
                              nms_iou_thr,
                              label_dict, 
                              category_index, 
                              save_path,
                              run_type='step'):
        """ Model inference. Results are saved in save_path.

            Args:
                sess (tf.Session())
                dataflow (Dataflow): dataflow for testing set
                im_rescale (int or list of two int): image size of model input
                obj_score_thr (float): confidence score threshold for selecting detected bbox
                nms_iou_thr (float): IoU threshold for non maximum suppression
                label_dict: a dict that maps integer ids to category names
                category_index: a dict that maps integer ids to category dicts. e.g.
                    {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
                save_path (str): path to save results
                run_type (str): type of inference ('step': inference for one batch;
                        'epoch': inference for one epoch)
        """

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
                        para,
                        score, 
                        n_class=self._n_class, 
                        obj_score_thr=obj_score_thr,
                        nms_iou_thr=nms_iou_thr, 
                        label_dict=label_dict,
                        image_shape=batch_data['shape'][idx], 
                        rescale_shape=im_rescale)

                original_im = imagetool.rescale_image(
                    batch_data['image'][idx] * 255, batch_data['shape'][idx])

                viz.draw_bounding_box_on_image_array(
                    original_im, 
                    bbox_list=nms_boxes, 
                    class_list=nms_label_ids,
                    score_list=nms_scores, 
                    category_index=category_index,
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

    # def _create_train_input(self):
    #     # self.o_shape = tf.placeholder(tf.float32, [None, 2], 'o_shape')
    #     self.image = tf.placeholder(
    #         tf.float32, [None, None, None, self._n_channel], name='image')
    #     self.label = tf.placeholder(
    #         tf.float32, [None, None, (1 + 4 + self._n_class)], 'label')
    #     # xyxy
    #     self.true_boxes = tf.placeholder(
    #         tf.float32, [None, None, 4], 'true_boxes')
    #     self.lr = tf.placeholder(tf.float32, name='lr')
    
    # def create_train_model(self):
    #     self.set_is_training(is_training=True)
    #     self._create_train_input()
    #     _, self.layers['pred_bbox'], self.layers['bbox_t_coord'], self.layers['objectness_logits'], self.layers['classes_logits']\
    #         = self._create_model(self.image)

    #     self.train_op = self.get_train_op()
    #     self.loss_op = self.get_loss()
    #     self.global_step = 0
    #     self.epoch_id = 0
    #     # self.layers['gt_mask'] = self._get_target_anchor(self._rescale_shape)

    # def train_epoch(self,
    #                 sess, 
    #                 train_data, 
    #                 pre_processor, 
    #                 init_lr, 
    #                 im_rescale,
    #                 summary_writer=None):
    #     """ Train the model for one epoch

    #         Args:
    #             sess (tf.Session())
    #             train_data (Dataflow): dataflow for training set
    #             pre_processor (PreProcess): object for computing target prediction and data augmentation
    #             init_lr (float): learning rate
    #             im_rescale (list of 2): image size of model input
    #             summary_writer (tf.summary)
    #     """

    #     display_name_list = ['cls_loss', 'bbox_loss', 'obj_loss', 'loss']
    #     cur_summary = None
    #     lr = init_lr

    #     cls_loss_sum = 0
    #     bbox_loss_sum = 0
    #     obj_loss_sum = 0
    #     loss_sum = 0
    #     step = 0

    #     self.epoch_id += 1
    #     cur_epoch = train_data.epochs_completed
    #     while train_data.epochs_completed == cur_epoch:
    #         step += 1
    #         self.global_step += 1
    #         # get one batch data
    #         # batch_data = train_data.next_batch_dict()

    #         im_batch, gt_mask_batch, true_boxes = pre_processor.process_batch(output_scale=im_rescale)

    #         # batch_gt, true_boxes = target_anchor.get_yolo_target_anchor(
    #         #     batch_data['label'], 
    #         #     batch_data['boxes'], 
    #         #     batch_data['shape'], 
    #         #     im_rescale, True)

    #         _, loss, cls_loss, bbox_loss, obj_loss = sess.run(
    #             [self.train_op, self.loss_op, self.cls_loss,
    #              self.bbox_loss, self.obj_loss],
    #             feed_dict={self.image: im_batch,
    #                        # self.o_shape: batch_data['shape'],
    #                        self.true_boxes: true_boxes,
    #                        self.label: gt_mask_batch,
    #                        self.lr: lr})

    #         cls_loss_sum += cls_loss
    #         bbox_loss_sum += bbox_loss
    #         obj_loss_sum += obj_loss
    #         loss_sum += loss
            
    #         if step % 10 == 0:
    #             viz.display(
    #                 self.global_step,
    #                 step,
    #                 [cls_loss_sum, bbox_loss_sum, obj_loss_sum, loss_sum],
    #                 display_name_list,
    #                 'train',
    #                 summary_val=cur_summary,
    #                 summary_writer=summary_writer)
    #     # write summary 
    #     print('==== epoch: {}, lr:{} ===='.format(cur_epoch, lr))
    #     viz.display(
    #         self.global_step,
    #         step,
    #         [cls_loss_sum, bbox_loss_sum, obj_loss_sum, loss_sum],
    #         display_name_list,
    #         'train',
    #         summary_val=cur_summary,
    #         summary_writer=summary_writer)


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

