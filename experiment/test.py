#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import platform
import numpy as np
import skimage.transform
import tensorflow as tf

import sys
sys.path.append('../')
import loader
import configs.parsecfg as parscfg

import src.utils.image as image
import src.dataflow.augmentation as augment
import src.evaluate.np_eval as np_eval
import src.bbox.bboxtool as bboxtool
from src.dataflow.preprocess import PreProcess
import src.dataflow.operation as dfoper


def test_target_anchor():
    import src.utils.viz as viz

    config = parscfg.ConfigParser('configs/{}_path.cfg'.format(platform.node()),
                                  'configs/coco80.cfg')

    image_data, label_dict, _ = loader.load_VOC(batch_size=2)
    rescale_shape = 320
    image_data.reset_image_rescale(rescale=rescale_shape)
    batch_data = image_data.next_batch_dict()
    gt_bbox_para = np.array([bbox[1:] for bbox in batch_data['label'][0]])
    gt_bbox_label = [bbox[0] for bbox in batch_data['label'][0]]

    stride_list = [32, 16, 8]
    target = bboxgt.TargetAnchor([416, 320], stride_list, config.anchors, config.n_class)
    gt, target_anchor_batch = target.get_yolo_target_anchor(
        batch_data['label'], batch_data['boxes'], batch_data['shape'], rescale_shape, True)

    # rescale_im = image.rescale_image(batch_data['image'][0]*255, rescale_shape)
    # o_im = image.rescale_image(batch_data['image'][0]*255, batch_data['shape'][0])
    # viz.draw_bounding_box(o_im, gt_bbox_para, label_list=None, box_type='xyxy')
    # viz.draw_bounding_box(rescale_im, target_anchor_batch[0], label_list=None, box_type='xyxy')

def test_mAP():
    pred_bboxes = [[25,35,45,55], [35,45,55,65],[250,350,450,550],[250,350,450,550],[250,350,450,550],
                    [45,65,55,75],[15,25,35,45],[250,350,450,550],[250,350,450,550],[35,25,55,45],[15,25,35,45],
                    ]
    gt_bboxes = [[25,35,45,55], [35,45,55,65],[45,65,55,75],[15,25,35,45],[35,25,55,45],[45,65,55,75],[15,25,35,45],[35,25,55,45]]
    pred_classes = [1,1,1,1,1,1,1,1,1,1, 2]
    gt_classes = [1,1,1,1,1,2,2,2]
    pred_conf = [0.9, 0.9, 0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1, 0.9]
    IoU_thr = 0.5
    pred_im_size = 1
    gt_im_size = 1

    re = np_eval.mAP(
        pred_bboxes, pred_classes, pred_conf, gt_bboxes,
        gt_classes, IoU_thr, pred_im_size, gt_im_size)

    print(re)

def test_IoU():
    b_1 = [[0, 2, 1, 3], [0, 2, 1, 3], [0, 5, 7, 10], [0, 0, 0, 0]] 
    b_2 = [[0, 2+5.5, 1+5.5, 3+5.5], [0, 2+0.5, 1+0.5, 3+0.5]]
    # b_2 = [i + 0.5 for i in b_2]
    # print(bbox_IOU(b_1, b_2, align=True))

    re = bboxtool.bbox_list_IOU(b_1, b_2, align=False)
    print(re)
    re = bboxtool.bbox_list_IOU(b_1, b_2, align=True)
    print(re)

def test_image_augment():
    import src.utils.viz as viz
    import matplotlib.pyplot as plt

    image_data, label_dict, _ = loader.load_VOC(batch_size=2)
    batch_data = image_data.next_batch_dict()

    im = batch_data['image'][0]
    bbox = batch_data['boxes'][0]

    # viz.draw_bounding_box(im*255, bbox, label_list=None, box_type='xyxy')

    # crop_im, c_bbox = augment.center_crop(im, [300, 400], bbox)
    # a_im, c_bbox = augment.crop(im, bbox, crop_size=[30, 30, 300, 300])
    # print(im.shape)
    a_im, c_bbox = augment.affine_transform(im, bbox, scale=[1., 1.], translation=[0.1, -0.1], shear=[0, 0.4], angle=10)
    # a_im = augment.change_color(im, hue=0, saturate=1, brightness=0.6, intensity_scale=1.)
    # a_im, c_bbox = augment.horizontal_flip(im, bbox)
    # c_bbox = augment.remove_invalid_bbox(a_im, c_bbox)

    a_im = augment.im_preserve_range(a_im, 1.)
    # print(a_im.shape)

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].imshow(im)
    ax[1].imshow(a_im)

    # # # plt.subplots(1,2,sharex='col')
    # # # plt.imshow(image.center_crop_image(im, [100, 200]))
    plt.show()

    
    # viz.draw_bounding_box(a_im*255, c_bbox, label_list=None, box_type='xyxy')

def test_preprocess():
    import time

    image_data, label_dict, _ = loader.load_VOC(batch_size=32)
    config = parscfg.ConfigParser('configs/{}_path.cfg'.format(platform.node()),
                                  'configs/coco80.cfg')

    stride_list = [32, 16, 8]

    pre = PreProcess(
        dataflow=image_data, 
        rescale_shape_list=[416, 320],
        stride_list=stride_list, 
        prior_list=config.anchors, 
        n_class=config.n_class,
        h_flip=True, 
        crop=True, 
        color=True, 
        affine=True,
        max_num_bbox_per_im=45)

    start_time = time.time()
    im, gt_mask_batch, true_boxes = pre.process_batch(output_scale=[320, 320])
    print(time.time() - start_time)

    start_time = time.time()
    pre.process_batch_2(output_scale=[320, 320])
    print(time.time() - start_time)

    # import src.utils.viz as viz
    # viz.draw_bounding_box(im[0]*255, true_boxes[0], label_list=None, box_type='xyxy')


def test_input():
    import time
    import src.utils.viz as viz

    config = parscfg.ConfigParser('configs/{}_path.cfg'.format(platform.node()),
                                  'configs/coco80.cfg')

    label_dict, category_index, train_data_generator, valid_data_generator = loader.load_VOC(
         rescale_shape_list=config.mutliscale,
         net_stride_list=[32, 16, 8], 
         prior_anchor_list=config.anchors,
         n_class=config.n_class,
         batch_size=1, 
         buffer_size=4,
         num_parallel_preprocess=2,
         h_flip=True, crop=True, color=True, affine=True,
         max_num_bbox_per_im=45)
    print(valid_data_generator.batch_data)

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        for epoch in range(10):
            print('epoch: {}'.format(epoch))
            train_data_generator.init_iterator(sess, reset_scale=True)
            while True:
                try:
                    # train_data_generator.init_iterator(sess, reset_scale=True)
                    # train_data_generator.reset_im_scale()
                    start_time = time.time()
                    t = sess.run(train_data_generator.batch_data)
                    print(time.time() - start_time)
                    print(t[0].shape)
                    print(t[-1].shape)
                    viz.draw_bounding_box(t[0][0]*255, t[2][0], label_list=None, box_type='xyxy')
                except tf.errors.OutOfRangeError:
                    break

# def test_divide_df():
#     import src.utils.viz as viz

#     config = parscfg.ConfigParser('configs/{}_path.cfg'.format(platform.node()),
#                                   'configs/coco80.cfg')

#     label_dict, category_index, train_data_generator = loader.load_VOC(
#          rescale_shape_list=config.mutliscale,
#          net_stride_list=[32, 16, 8], 
#          prior_anchor_list=config.anchors,
#          n_class=config.n_class,
#          batch_size=config.train_bsize, 
#          buffer_size=4,
#          num_parallel_preprocess=8,
#          h_flip=True, crop=True, color=True, affine=True,
#          max_num_bbox_per_im=45)

    # df_list = dfoper.divide_dataflow(train_data, divide_list=[0.2, 0.3, 0.2], shuffle=True)
    # for df in df_list:
    #     print(df.size())


    # batch_data = df_list[-1].next_batch_dict()
    # print(batch_data['label'][0][1:])
    # box_batch = [bbox[1:] for bbox in batch_data['label'][0]]
    # viz.draw_bounding_box(batch_data['image'][0]*255, box_batch, label_list=None, box_type='xyxy')

if __name__ == "__main__":
    test_input()
        
