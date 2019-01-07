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

    # pathconfig = parscfg.parse_cfg('configs/{}_path.cfg'.format(platform.node()))
    # pretrained_path = pathconfig['yolo_feat_pretraind_npy']
    # data_dir = pathconfig['test_image_path']
    # save_path = pathconfig['save_path']
    # im_name = pathconfig['test_image_name']

    # netconfig = parscfg.parse_cfg('configs/voc.cfg')
    # im_rescale = netconfig['rescale']
    # mutliscale = netconfig['multiscale']
    # n_channel = netconfig['n_channel']
    # test_bsize = netconfig['test_bsize']
    # train_bsize = netconfig['train_bsize']
    # obj_score_thr = netconfig['obj_score_thr']
    # nms_iou_thr = netconfig['nms_iou_thr']
    # n_class = netconfig['n_class']
    # anchors = netconfig['anchors']
    # ignore_thr = netconfig['ignore_thr']

    config = parscfg.ConfigParser('configs/{}_path.cfg'.format(platform.node()),
                                  'configs/voc.cfg')
    im_rescale = config.im_rescale
    
    # Training
    train_data, label_dict, category_index = loader.load_VOC(
        batch_size=config.train_bsize, rescale=im_rescale)
    target_anchor = TargetAnchor(
        config.mutliscale,
        [32, 16, 8], 
        config.anchors, 
        config.n_class, 
        ignore_thr=config.ignore_thr)

    train_model = YOLOv3(
        n_channel=config.n_channel,
        n_class=config.n_class, 
        anchors=config.anchors,
        bsize=config.train_bsize, 
        ignore_thr=config.ignore_thr,
        obj_weight=config.obj_weight,
        nobj_weight=config.nobj_weight,
        feature_extractor_trainable=False, 
        detector_trainable=True,
        pre_trained_path=config.yolo_feat_pretrained_path,)
    train_model.create_train_model()
    
    # Validation
    # label_dict, category_index = loader.load_coco80_label_yolo()
    test_scale = 416
    image_data = loader.read_image(
        im_name=config.im_name, 
        n_channel=config.n_channel,
        data_dir=config.data_dir, 
        batch_size=config.test_bsize, 
        rescale=test_scale)
    test_model = YOLOv3(
        n_channel=config.n_channel, 
        n_class=config.n_class, 
        anchors=config.anchors,
        bsize=config.test_bsize, 
        # obj_score_thr=config.obj_score_thr, 
        # nms_iou_thr=config.nms_iou_thr,
        feature_extractor_trainable=False, 
        detector_trainable=False,
        pre_trained_path=config.yolo_feat_pretrained_path,)
    test_model.create_valid_model()

    train_op = train_model.get_train_op()
    loss_op = train_model.get_loss()

    writer = tf.summary.FileWriter(config.save_path)
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
                sess,
                image_data, 
                test_scale, 
                config.obj_score_thr, 
                config.nms_iou_thr,
                label_dict, 
                category_index, 
                config.save_path, 
                run_type='epoch')
            
            train_model.train_epoch(
                sess, 
                train_data, 
                target_anchor, 
                lr, 
                im_rescale,
                summary_writer=writer)

            if i > 0 and i % 10 == 0:
                pick_id = np.random.choice(len(config.mutliscale))
                im_rescale = config.mutliscale[pick_id]
                train_data.reset_image_rescale(rescale=im_rescale)
                print('rescale to {}'.format(im_rescale))

def detect():
    # pathconfig = parscfg.parse_cfg('configs/{}_path.cfg'.format(platform.node()))
    # pretrained_path = pathconfig['coco_pretrained_npy_path']
    # save_path = pathconfig['save_path']
    # data_dir = pathconfig['test_image_path']
    # im_name = pathconfig['test_image_name']

    # netconfig = parscfg.parse_cfg('configs/coco80.cfg')
    # im_rescale = netconfig['rescale']
    # n_channel = netconfig['n_channel']
    # bsize = netconfig['test_bsize']
    # obj_score_thr = netconfig['obj_score_thr']
    # nms_iou_thr = netconfig['nms_iou_thr']
    # n_class = netconfig['n_class']
    # anchors = netconfig['anchors']

    config = parscfg.ConfigParser('configs/{}_path.cfg'.format(platform.node()),
                                  'configs/coco80.cfg')

    label_dict, category_index = loader.load_coco80_label_yolo()
    # Create a Dataflow object for test images
    image_data = loader.read_image(
        im_name=config.im_name,
        n_channel=config.n_channel,
        data_dir=config.data_dir,
        batch_size=config.test_bsize, 
        rescale=config.im_rescale)

    test_model = YOLOv3(
        bsize=config.test_bsize,
        n_channel=config.n_channel,
        n_class=config.n_class, 
        anchors=config.anchors,
        # obj_score_thr=config.obj_score_thr, 
        # nms_iou_thr=config.nms_iou_thr,
        feature_extractor_trainable=False, 
        detector_trainable=False,
        pre_trained_path=config.coco_pretrained_path,)
    test_model.create_valid_model()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())

        test_model.predict_epoch_or_step(
            sess,
            image_data,
            config.im_rescale,
            config.obj_score_thr, 
            config.nms_iou_thr,
            label_dict, 
            category_index, 
            config.save_path, 
            run_type='epoch')

if __name__ == '__main__':
    FLAGS = get_args()

    if FLAGS.train:
        train()
    if FLAGS.detect:
        detect()

    # config = parscfg.ConfigParser('configs/{}_path.cfg'.format(platform.node()), 'configs/voc.cfg')
    # print(config.test_bsize)
    # def test1():
    #     p = 0
    #     def test():
    #         nonlocal p
    #         print(p)
    #         p+=1

    #     test()
    #     test()
    # test1()

