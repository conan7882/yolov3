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
from src.dataflow.voc import VOC, get_class_dict_from_xml


def load_VOC(batch_size=1):
    if platform.node() == 'arostitan':
        raise ValueError('Data path does not setup on this platform!')
        # data_path = '/home/qge2/workspace/data/foram/CNN_sythetic/edge_set/'
    elif platform.node() == 'aros04':
        im_dir = 'E:/Dataset/VOCdevkit/VOC2007/JPEGImages/'
        xml_dir = 'E:/Dataset/VOCdevkit/VOC2007/Annotations/'
    else:
        im_dir = '/Users/gq/workspace/Dataset/VOCdevkit/VOC2007/JPEGImages/'
        xml_dir = '/Users/gq/workspace/Dataset/VOCdevkit/VOC2007/Annotations/'

    class_dict, reverse_class_dict = get_class_dict_from_xml(xml_dir)

    train_data = VOC(
        class_dict=class_dict,
        image_dir=im_dir,
        xml_dir=xml_dir,
        n_channel=3,
        shuffle=True,
        batch_dict_name=['image', 'label'],
        pf_list=None)
    train_data.setup(epoch_val=0, batch_size=batch_size)

    return train_data

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

    def rescale_im(im, short_side=224):
        """ Pre-process for images 
            images are rescaled so that the shorter side = 224
        """
        im = np.array(im)
        h, w = im.shape[0], im.shape[1]
        if h >= w:
            new_w = short_side
            im = skimage.transform.resize(
                im, (int(h * new_w / w), short_side), preserve_range=True)
        else:
            new_h = short_side
            im = skimage.transform.resize(
                im, (short_side, int(w * new_h / h)), preserve_range=True)
        # return im.astype('uint8')
        return im

    def normalize_im(im):
        if rescale is not None:
            im = rescale_im(im, short_side=rescale)
        im = np.array(im)
        if np.amax(im) > 1:
            im = im / 255.
        return np.clip(im, 0., 1.)

    # if rescale:
    #     pf_fnc = rescale_im
    # else:
    #     pf_fnc = normalize_im

    image_data = Image(
        im_name=im_name,
        data_dir=data_dir,
        n_channel=n_channel,
        shuffle=False,
        batch_dict_name=['image'],
        pf_list=normalize_im)
    image_data.setup(epoch_val=0, batch_size=batch_size)

    return image_data

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import src.utils.viz as viz

    xml_dir = '/Users/gq/workspace/Dataset/VOCdevkit/VOC2007/Annotations/'
    # 
    # print(class_dict, reverse_class_dict)
    dataflow = load_VOC(batch_size=2)
    # print(dataflow._file_name_list)

    batch_date = dataflow.next_batch_dict()

    print(batch_date['label'])
    print([[bbox[0] for bbox in bbox_list] for bbox_list in batch_date['label']])
    box_list = [[bbox[1] for bbox in bbox_list] for bbox_list in batch_date['label']]
    print(box_list)

    viz.draw_bounding_box(batch_date['image'][0], box_list[0])
    # plt.figure()
    # plt.imshow(batch_date['image'][0])
    # plt.show()

