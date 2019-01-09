#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: yolov3.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import argparse
import platform
import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
import loader
import configs.parsecfg as parscfg
from src.net.yolov3 import YOLOv3


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true',
                        help='Train the model.')
    parser.add_argument('--detect', action='store_true',
                        help='detect')

    parser.add_argument('--lr', type=float, default=0.1,
                        help='Batch size')
    
    return parser.parse_args()

def train():
    FLAGS = get_args()
    config = parscfg.ConfigParser('configs/{}_path.cfg'.format(platform.node()),
                                  'configs/voc.cfg')

    # train_data, label_dict, category_index = loader.load_VOC(
    #     batch_size=config.train_bsize)

    # gen = generator.Generator(
    #     dataflow=train_data, 
    #     n_channle=config.n_channel,
    #     rescale_shape_list=config.mutliscale,
    #     stride_list=[32, 16, 8], 
    #     prior_list=config.anchors, 
    #     n_class=config.n_class,
    #     batch_size=config.train_bsize, 
    #     buffer_size=4, 
    #     num_parallel_preprocess=8,
    #     h_flip=True, crop=True, color=True, affine=True, im_intensity = 1.,
    #     max_num_bbox_per_im=45)
    # gen.reset_im_scale(scale=416)

    label_dict, category_index, train_data_generator = loader.load_VOC(
         rescale_shape_list=config.mutliscale,
         net_stride_list=[32, 16, 8], 
         prior_anchor_list=config.anchors,
         n_class=config.n_class,
         batch_size=config.train_bsize, 
         buffer_size=4,
         num_parallel_preprocess=8,
         h_flip=True, crop=True, color=True, affine=True,
         max_num_bbox_per_im=45)

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
    train_model.create_train_model(train_data_generator.batch_data)

    # Validation
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
        feature_extractor_trainable=False, 
        detector_trainable=False,
        pre_trained_path=config.yolo_feat_pretrained_path,)
    test_model.create_valid_model()

    writer = tf.summary.FileWriter(config.save_path)
    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(150*3):
            if i >= 100*3:
                lr = FLAGS.lr / 100.
            elif i >= 50*2:
                lr = FLAGS.lr / 10.
            else:
                lr = FLAGS.lr

            train_data_generator.init_iterator(sess)

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
            
            train_model.train_epoch(sess, lr, summary_writer=writer)

            if i > 0 and i % 20 == 0:
                train_data_generator.reset_im_scale()

# def train():
#     FLAGS = get_args()
#     config = parscfg.ConfigParser('configs/{}_path.cfg'.format(platform.node()),
#                                   'configs/voc.cfg')
#     train_im_rescale = config.im_rescale
    
#     # Training
#     train_data, label_dict, category_index = loader.load_VOC(
#         batch_size=config.train_bsize, rescale=train_im_rescale)
#     # target_anchor = TargetAnchor(
#     #     config.mutliscale,
#     #     [32, 16, 8], 
#     #     config.anchors, 
#     #     config.n_class, 
#     #     ignore_thr=config.ignore_thr)

#     pre_processor = PreProcess(
#         dataflow=train_data, 
#         rescale_shape_list=config.mutliscale,
#         stride_list=[32, 16, 8],
#         prior_list=config.anchors, 
#         n_class=config.n_class,
#         h_flip=True, 
#         crop=True, 
#         color=True, 
#         affine=True,
#         im_intensity=1.,
#         max_num_bbox_per_im=45)

#     train_model = YOLOv3(
#         n_channel=config.n_channel,
#         n_class=config.n_class, 
#         anchors=config.anchors,
#         bsize=config.train_bsize, 
#         ignore_thr=config.ignore_thr,
#         obj_weight=config.obj_weight,
#         nobj_weight=config.nobj_weight,
#         feature_extractor_trainable=False, 
#         detector_trainable=True,
#         pre_trained_path=config.yolo_feat_pretrained_path,)
#     train_model.create_train_model()
    
#     # Validation
#     # label_dict, category_index = loader.load_coco80_label_yolo()
#     test_scale = 416
#     image_data = loader.read_image(
#         im_name=config.im_name, 
#         n_channel=config.n_channel,
#         data_dir=config.data_dir, 
#         batch_size=config.test_bsize, 
#         rescale=test_scale)
#     test_model = YOLOv3(
#         n_channel=config.n_channel, 
#         n_class=config.n_class, 
#         anchors=config.anchors,
#         bsize=config.test_bsize, 
#         # obj_score_thr=config.obj_score_thr, 
#         # nms_iou_thr=config.nms_iou_thr,
#         feature_extractor_trainable=False, 
#         detector_trainable=False,
#         pre_trained_path=config.yolo_feat_pretrained_path,)
#     test_model.create_valid_model()

#     writer = tf.summary.FileWriter(config.save_path)
#     sessconfig = tf.ConfigProto()
#     sessconfig.gpu_options.allow_growth = True
#     with tf.Session(config=sessconfig) as sess:
#         sess.run(tf.global_variables_initializer())

#         for i in range(150*3):
#             if i >= 100*3:
#                 lr = FLAGS.lr / 100.
#             elif i >= 50*2:
#                 lr = FLAGS.lr / 10.
#             else:
#                 lr = FLAGS.lr

#             test_model.predict_epoch_or_step(
#                 sess,
#                 image_data, 
#                 test_scale, 
#                 config.obj_score_thr, 
#                 config.nms_iou_thr,
#                 label_dict, 
#                 category_index, 
#                 config.save_path, 
#                 run_type='epoch')
            
#             train_model.train_epoch(
#                 sess, 
#                 train_data, 
#                 pre_processor, 
#                 lr, 
#                 train_im_rescale,
#                 summary_writer=writer)

#             if i > 0 and i % 20 == 0:
#                 pick_id = np.random.choice(len(config.mutliscale))
#                 train_im_rescale = config.mutliscale[pick_id]
#                 train_im_rescale = [train_im_rescale, train_im_rescale]
#                 print('rescale to {}'.format(train_im_rescale))

def detect():
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

