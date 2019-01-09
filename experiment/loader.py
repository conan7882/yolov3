#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: loader.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import platform
import numpy as np
import skimage.transform

import sys
sys.path.append('../')
from src.dataflow.images import Image
from src.dataflow.voc import VOC
from src.utils.dataflow import get_class_dict_from_xml
import src.utils.image as imagetool
import src.dataflow.generator as generator


def load_coco80_label_yolo():
    label_dict = {}
    file_path = '../data/coco.names'
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            names = line.rstrip()
            label_dict[idx] = names

    category_index = {}
    for class_id in label_dict:
        category_index[class_id] = {'id': class_id, 'name': label_dict[class_id]}

    return label_dict, category_index

def load_imagenet1k_label_darknet():
    """ 
        Function to read the ImageNet label file.
        Used for testing the pre-trained model.

        dataset (str): name of data set. 'imagenet', 'cifar'
    """
    imagenet_label_dict = {}
    file_path = '../data/imageNetLabel.txt'
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            names = line.rstrip()[10:]
            imagenet_label_dict[line.rstrip()[:9]] = names

    label_dict = {}
    file_path = '../data/imagenet.labels.list'
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= 1000:
                break
            label_dict[idx] = imagenet_label_dict[line.rstrip('\n')]
    return label_dict

def load_VOC(rescale_shape_list,
             net_stride_list, 
             prior_anchor_list, 
             n_class=None, 
             max_num_bbox_per_im=45,
             batch_size=1, 
             buffer_size=4,
             num_parallel_preprocess=8,
             h_flip=True, crop=True, color=True, affine=True):

    if platform.node() == 'arostitan':
        im_dir = '/home/qge2/workspace/data/dataset/VOCdevkit/VOC2007/JPEGImages/'
        xml_dir = '/home/qge2/workspace/data/dataset/VOCdevkit/VOC2007/Annotations/'
    elif platform.node() == 'aros04':
        im_dir = 'E:/Dataset/VOCdevkit/VOC2007/JPEGImages/'
        xml_dir = 'E:/Dataset/VOCdevkit/VOC2007/Annotations/'
    else:
        im_dir = '/Users/gq/workspace/Dataset/VOCdevkit/VOC2007/JPEGImages/'
        xml_dir = '/Users/gq/workspace/Dataset/VOCdevkit/VOC2007/Annotations/'

    class_name_dict, class_id_dict = get_class_dict_from_xml(xml_dir)

    category_index = {}
    for class_id in class_id_dict:
        category_index[class_id] = {'id': class_id, 'name': class_id_dict[class_id]}

    def normalize_im(im, *args):
        # im = imagetool.rescale_image(im, args[0])
        im = np.array(im)
        if np.amax(im) > 1:
            im = im / 255.
        return np.clip(im, 0., 1.)

    train_data = VOC(
        class_dict=class_name_dict,
        image_dir=im_dir,
        xml_dir=xml_dir,
        n_channel=3,
        shuffle=True,
        batch_dict_name=['image', 'label', 'shape'],
        pf_list=(normalize_im, ())
        )
    train_data.setup(epoch_val=0, batch_size=1)

    if n_class is None:
        n_class = len(class_name_dict)

    data_generator = generator.Generator(
        dataflow=train_data, 
        n_channle=3,
        rescale_shape_list=rescale_shape_list,
        stride_list=net_stride_list, 
        prior_list=prior_anchor_list, 
        n_class=n_class,
        batch_size=batch_size, 
        buffer_size=buffer_size, 
        num_parallel_preprocess=num_parallel_preprocess,
        h_flip=h_flip, crop=crop, color=color, affine=affine, im_intensity = 1.,
        max_num_bbox_per_im=max_num_bbox_per_im)
    data_generator.reset_im_scale(scale=416)

    return class_id_dict, category_index, data_generator

def read_image(im_name, n_channel, data_dir='', batch_size=1, rescale=None):
    """ function for create a Dataflow for reading images from a folder
        This function returns a Dataflow object for images with file 
        name containing 'im_name' in directory 'data_dir'.

        Args:
            im_name (str): part of image names (i.e. 'jpg' or 'im_').
            n_channel (int): number of channels (3 for color images and 1 for grayscale images)
            data_dir (str): directory of images
            batch_size (int): number of images read from Dataflow for each batch
            rescale (bool): whether rescale image to 224 or not

        Returns:
            Image (object): batch images can be access by Image.next_batch_dict()['image']
    """

    def rescale_im(im, short_side=416):
        """ Pre-process for images 
            images are rescaled so that the shorter side = 224
        """
        im = np.array(im)
        h, w = im.shape[0], im.shape[1]
        if h >= w:
            new_w = short_side
            im = imagetool.rescale_image(im, (int(h * new_w / w), short_side))
            # im = skimage.transform.resize(
            #     im, (int(h * new_w / w), short_side), preserve_range=True)
        else:
            new_h = short_side
            im = imagetool.rescale_image(im, (short_side, int(w * new_h / h)))
            # im = skimage.transform.resize(
            #     im, (short_side, int(w * new_h / h)), preserve_range=True)
        # return im.astype('uint8')
        return im

    def normalize_im(im, *args):
        im = imagetool.rescale_image(im, rescale)
        # im = skimage.transform.resize(
        #     im, rescale, preserve_range=True)
        #     im = rescale_im(im, short_side=rescale)
        im = np.array(im)
        if np.amax(im) > 1:
            im = im / 255.
        return np.clip(im, 0., 1.)

    # if rescale:
    #     pf_fnc = rescale_im
    # else:
    #     pf_fnc = normalize_im

    if isinstance(rescale, int):
        rescale = [rescale, rescale]
    else:
        assert len(rescale) == 2

    image_data = Image(
        im_name=im_name,
        data_dir=data_dir,
        n_channel=n_channel,
        shuffle=False,
        batch_dict_name=['image', 'shape'],
        pf_list=(normalize_im,()))
    image_data.setup(epoch_val=0, batch_size=batch_size)

    return image_data
