#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: yolov3.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import argparse
import platform
import numpy as np
import tensorflow as tf
# import skimage.transform

import sys
sys.path.append('../')
import loader
import configs.parsecfg as parscfg
from src.net.yolov3 import YOLOv3
import src.utils.viz as viz
from src.bbox.py_nms import non_max_suppression
import src.model.detection_bbox as detection_bbox
import src.utils.image as imagetool
from src.bbox.bboxgt import TargetAnchor


# PRETRINED_PATH = '/Users/gq/workspace/Dataset/pretrained/yolov3.npy'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true',
                        help='Train the model.')
    parser.add_argument('--detect', action='store_true',
                        help='detect')

    # parser.add_argument('--pretrained_path', type=str, default=PRETRINED_PATH,
    #                     help='Directory of pretrain model')
    # parser.add_argument('--im_name', type=str, default='.jpg',
    #                     help='Part of image name')
    # parser.add_argument('--data_dir', type=str, default='../data/',
    #                     help='Directory of test images')

    # parser.add_argument('--rescale', type=int, default=416,
    #                     help='Rescale input image with shorter side = rescale')
    # parser.add_argument('--bsize', type=int, default=2,
    #                     help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Batch size')
    
    return parser.parse_args()

def train():
    FLAGS = get_args()

    pathconfig = parscfg.parse_cfg('configs/{}_path.cfg'.format(platform.node()))
    pretrained_path = pathconfig['yolo_feat_pretraind_npy']
    data_dir = pathconfig['test_image_path']
    save_path = pathconfig['save_path']
    im_name = pathconfig['test_image_name']

    netconfig = parscfg.parse_cfg('configs/voc.cfg')
    im_rescale = netconfig['rescale']
    mutliscale = netconfig['multiscale']
    n_channel = netconfig['n_channel']
    test_bsize = netconfig['test_bsize']
    train_bsize = netconfig['train_bsize']
    obj_score_thr = netconfig['obj_score_thr']
    nms_iou_thr = netconfig['nms_iou_thr']
    n_class = netconfig['n_class']
    anchors = netconfig['anchors']
    
    # Training
    train_data, label_dict, category_index = loader.load_VOC(batch_size=train_bsize, rescale=im_rescale)
    target_anchor = TargetAnchor(
        [im_rescale], [32, 16, 8], anchors, n_class)
    train_model = YOLOv3(
        n_channel=n_channel, n_class=n_class, anchors=anchors,
        rescale_shape=im_rescale, pre_trained_path=pretrained_path,
        bsize=train_bsize, obj_score_thr=obj_score_thr, nms_iou_thr=nms_iou_thr,
        feature_extractor_trainable=False, detector_trainable=True)
    train_model.create_train_model()
    
    # Validation
    # label_dict, category_index = loader.load_coco80_label_yolo()
    image_data = loader.read_image(
        im_name=im_name, n_channel=n_channel,
        data_dir=data_dir, batch_size=test_bsize, rescale=im_rescale)
    test_model = YOLOv3(
        n_channel=n_channel, n_class=n_class, anchors=anchors,
        rescale_shape=im_rescale, pre_trained_path=pretrained_path,
        bsize=test_bsize, obj_score_thr=obj_score_thr, nms_iou_thr=nms_iou_thr,
        feature_extractor_trainable=False, detector_trainable=False)
    test_model.create_valid_model()

    train_op = train_model.get_train_op()
    loss_op = train_model.get_loss()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10):
            test_model.predict_epoch_or_step(
                sess, image_data, im_rescale, obj_score_thr, nms_iou_thr,
                label_dict, category_index, save_path, run_type='step')
            # print('epoch: {}'.format(i))
            # batch_data = image_data.next_batch_dict()
            # bbox_score, bbox_para = sess.run(
            #     [test_model.layers['bbox_score'], test_model.layers['bbox']],
            #     feed_dict={test_model.image: batch_data['image']})
            # for idx, (score, para) in enumerate(zip(bbox_score, bbox_para)):
            #     nms_boxes, nms_scores, nms_label_names, nms_label_ids =\
            #         detection_bbox.detection(
            #             para, score, n_class=n_class, obj_score_thr=obj_score_thr,
            #             nms_iou_thr=nms_iou_thr, label_dict=label_dict,
            #             image_shape=batch_data['shape'][idx], rescale_shape=im_rescale)
            #     original_im = imagetool.rescale_image(
            #         batch_data['image'][idx] * 255, batch_data['shape'][idx])
            #     viz.draw_bounding_box_on_image_array(
            #         original_im, bbox_list=nms_boxes, class_list=nms_label_ids,
            #         score_list=nms_scores, category_index=category_index,
            #         save_name=os.path.join(save_path, 'im_{}.png'.format(idx)),
            #         save_fig=True)
            #     print(nms_label_names)
            
            train_model.train_epoch(
                sess, train_data, target_anchor, FLAGS.lr, im_rescale,
                summary_writer=None)
            # cur_epoch = train_data.epochs_completed
            # step = 0
            # while train_data.epochs_completed == cur_epoch:
            #     step += 1
            #     batch_data = train_data.next_batch_dict()
            #     batch_gt = target_anchor.get_yolo_target_anchor(
            #         batch_data['label'], batch_data['shape'], im_rescale, True)

            #     _, loss = sess.run(
            #         [train_op, loss_op],
            #         feed_dict={train_model.image: batch_data['image'],
            #                    train_model.o_shape: batch_data['shape'],
            #                    train_model.label: batch_gt,
            #                    train_model.lr: FLAGS.lr})
            #     print('step: {}, loss: {}'.format(step, loss))
            # break

def detect():
    pathconfig = parscfg.parse_cfg('configs/{}_path.cfg'.format(platform.node()))
    pretrained_path = pathconfig['coco_pretrained_npy_path']
    save_path = pathconfig['save_path']
    data_dir = pathconfig['test_image_path']
    im_name = pathconfig['test_image_name']

    netconfig = parscfg.parse_cfg('configs/coco80.cfg')
    im_rescale = netconfig['rescale']
    n_channel = netconfig['n_channel']
    bsize = netconfig['test_bsize']
    obj_score_thr = netconfig['obj_score_thr']
    nms_iou_thr = netconfig['nms_iou_thr']
    n_class = netconfig['n_class']
    anchors = netconfig['anchors']

    label_dict, category_index = loader.load_coco80_label_yolo()
    # Create a Dataflow object for test images
    image_data = loader.read_image(
        im_name=im_name, n_channel=n_channel,
        data_dir=data_dir, batch_size=bsize, rescale=im_rescale)

    test_model = YOLOv3(
        n_channel=n_channel, n_class=n_class, anchors=anchors,
        rescale_shape=im_rescale, pre_trained_path=pretrained_path,
        bsize=bsize, obj_score_thr=obj_score_thr, nms_iou_thr=nms_iou_thr,
        feature_extractor_trainable=False, detector_trainable=False)
    test_model.create_valid_model()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        im_id = 0
        while image_data.epochs_completed < 1:
            batch_data = image_data.next_batch_dict()
            # get batch file names
            batch_file_name = image_data.get_batch_file_name()[0]
            # get prediction results
            # det_score, det_bbox, det_class = sess.run(
            #     [test_model.layers['det_score'], test_model.layers['det_bbox'],
            #      test_model.layers['det_class']],
            #     feed_dict={test_model.image: batch_data['image'],
            #                test_model.o_shape: batch_data['shape']})

            # for idx, (score, para, class_id) in enumerate(zip(det_score, det_bbox, det_class)):
            #     nms_labels = [label_dict[i] for i in class_id]
            #     viz.draw_bounding_box_on_image_array(
            #         skimage.transform.resize(
            #             batch_data['image'][idx] * 255,
            #             batch_data['shape'][idx],
            #             preserve_range=True),
            #         bbox_list=para, class_list=list(map(int, class_id)),
            #         score_list=score, category_index=category_index)

            bbox_score, bbox_para = sess.run(
                [test_model.layers['bbox_score'], test_model.layers['bbox']],
                feed_dict={test_model.image: batch_data['image']})

            for idx, (score, para) in enumerate(zip(bbox_score, bbox_para)):
                nms_boxes, nms_scores, nms_label_names, nms_label_ids = detection_bbox.detection(
                    para, score, n_class=n_class, obj_score_thr=obj_score_thr,
                    nms_iou_thr=nms_iou_thr, label_dict=label_dict,
                    image_shape=batch_data['shape'][idx], rescale_shape=im_rescale)
                print(im_id)
                # original_im = imagetool.rescale_image(
                #     batch_data['image'][idx] * 255, batch_data['shape'][idx])
                # viz.draw_bounding_box_on_image_array(
                #     original_im, bbox_list=nms_boxes, class_list=nms_label_ids,
                #     score_list=nms_scores, category_index=category_index,
                #     save_name=os.path.join(save_path, 'im_{}.png'.format(im_id)),
                #     save_fig=True)
                im_id += 1

if __name__ == '__main__':
    FLAGS = get_args()

    if FLAGS.train:
        train()
    if FLAGS.detect:
        detect()
    
