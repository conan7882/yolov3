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
    parser.add_argument('--lr', type=float, default=0.1,
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
    ignore_thr = netconfig['ignore_thr']
    
    # Training
    train_data, label_dict, category_index = loader.load_VOC(batch_size=train_bsize, rescale=im_rescale)
    target_anchor = TargetAnchor(
        mutliscale, [32, 16, 8], anchors, n_class, ignore_thr=ignore_thr)
    train_model = YOLOv3(
        n_channel=n_channel, n_class=n_class, anchors=anchors,
        pre_trained_path=pretrained_path,
        bsize=train_bsize, obj_score_thr=obj_score_thr, nms_iou_thr=nms_iou_thr,
        feature_extractor_trainable=False, detector_trainable=True)
    train_model.create_train_model()
    
    # Validation
    # label_dict, category_index = loader.load_coco80_label_yolo()
    test_scale = 416
    image_data = loader.read_image(
        im_name=im_name, n_channel=n_channel,
        data_dir=data_dir, batch_size=test_bsize, rescale=test_scale)
    test_model = YOLOv3(
        n_channel=n_channel, n_class=n_class, anchors=anchors,
        pre_trained_path=pretrained_path,
        bsize=test_bsize, obj_score_thr=obj_score_thr, nms_iou_thr=nms_iou_thr,
        feature_extractor_trainable=False, detector_trainable=False)
    test_model.create_valid_model()

    train_op = train_model.get_train_op()
    loss_op = train_model.get_loss()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(150):
            if i >= 100:
                lr = FLAGS.lr / 100.
            elif i >= 50:
                lr = FLAGS.lr / 10.
            else:
                lr = FLAGS.lr

            test_model.predict_epoch_or_step(
                sess, image_data, test_scale, obj_score_thr, nms_iou_thr,
                label_dict, category_index, save_path, run_type='epoch')
            
            train_model.train_epoch(
                sess, train_data, target_anchor, lr, im_rescale,
                summary_writer=None)

            if i > 0 and i % 10 == 0:
                pick_id = np.random.choice(len(mutliscale))
                im_rescale = mutliscale[pick_id]
                train_data.reset_image_rescale(rescale=im_rescale)
                print('rescale to {}'.format(im_rescale))

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
        im_name=im_name,
        n_channel=n_channel,
        data_dir=data_dir,
        batch_size=bsize, 
        rescale=im_rescale)

    test_model = YOLOv3(
        bsize=bsize,
        n_channel=n_channel,
        n_class=n_class, 
        anchors=anchors,
        obj_score_thr=obj_score_thr, 
        nms_iou_thr=nms_iou_thr,
        feature_extractor_trainable=False, 
        detector_trainable=False,
        pre_trained_path=pretrained_path,)
    test_model.create_valid_model()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())

        test_model.predict_epoch_or_step(
            sess,
            image_data,
            im_rescale,
            obj_score_thr, 
            nms_iou_thr,
            label_dict, 
            category_index, 
            save_path, 
            run_type='epoch')

if __name__ == '__main__':
    FLAGS = get_args()

    if FLAGS.train:
        train()
    if FLAGS.detect:
        detect()
    
    # def test1():
    #     p = 0
    #     def test():
    #         nonlocal p
    #         print(p)
    #         p+=1

    #     test()
    #     test()
    # test1()

