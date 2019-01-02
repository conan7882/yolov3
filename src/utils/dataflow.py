#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dataflow.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import imageio
import numpy as np
from datetime import datetime
import xml.etree.ElementTree as ET
import src.utils.utils as utils


def identity(inputs, *args):
    return inputs

def load_image(im_path, read_channel=None, pf=(identity, ())):
    """ Load one image from file and apply pre-process function.

    Args:
        im_path (str): directory of image
        read_channel (int): number of image channels. Image will be read
            without channel information if ``read_channel`` is None.
        pf: pre-process fucntion

    Return:
        image after pre-processed with size [heigth, width, channel]

    """

    if read_channel is None:
        im = imageio.imread(im_path)
    elif read_channel == 3:
        im = imageio.imread(im_path, as_gray=False, pilmode="RGB")
    else:
        im = imageio.imread(im_path, as_gray=True)

    if len(im.shape) < 3:
        im = pf[0](im, pf[1])
        im = np.reshape(im, [im.shape[0], im.shape[1], 1])
    else:
        im = pf[0](im, pf[1])

    return im

def parse_bbox_xml(xml_path, class_dict=None, pf=(identity,())):
    """
        Returns:
            [class_name, [xmin, ymin, xmax, ymax]]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    box_list = []
    for obj in root.findall('object'):
        name = obj.find('name').text

        box = obj.find('bndbox')
        xmin = float(box.find('xmin').text)
        ymin = float(box.find('ymin').text)
        xmax = float(box.find('xmax').text)
        ymax = float(box.find('ymax').text)
        box = [xmin, ymin, xmax, ymax]
        box = pf[0](box, pf[1])
        # box_list.append(box)
        try:
            # box_list.append((class_dict[name], box))
            box_list.append(np.array([class_dict[name],] + box))
        except TypeError:
            # box_list.append((name, box))
            box_list.append([name,] + box)
    return box_list

def get_class_dict_from_xml(xml_path):
    file_list = get_file_list(xml_path, 'xml')
    class_dict = {}
    reverse_class_dict = {}
    nclass = 0
    for xml_file in file_list:
        bbox_list = parse_bbox_xml(xml_file)
        for bbox in bbox_list:
            cur_name = bbox[0]
            if cur_name not in class_dict:
                class_dict[cur_name] = nclass
                reverse_class_dict[nclass] = cur_name
                nclass += 1

    return class_dict, reverse_class_dict

def get_voc_bbox(xml_path):
    bboxs = []
    file_list = get_file_list(xml_path, 'xml')
    class_dict = {}
    reverse_class_dict = {}
    nclass = 0
    for xml_file in file_list:
        bbox_list = parse_bbox_xml(xml_file)
        bbox_list = [bbox[1:] for bbox in bbox_list]
        bboxs.extend(bbox_list)

    return bboxs

def vec2onehot(vec, n_class):
    vec = np.array(vec)
    one_hot = np.zeros((len(vec), n_class))
    one_hot[np.arange(len(vec)), vec] = 1
    return one_hot

def fill_pf_list(pf_list, n_pf, fill_with_fnc=(identity,())):
    """ Fill the pre-process function list.

    Args:
        pf_list (list): input list of pre-process functions
        n_pf (int): required number of pre-process functions 
        fill_with_fnc: function used to fill the list

    Returns:
        list of pre-process function
    """
    if pf_list == None:
        return [fill_with_fnc for i in range(n_pf)]

    new_list = []
    pf_list = utils.make_list(pf_list)
    for pf in pf_list:
        if not pf:
            pf = fill_with_fnc
        new_list.append(pf)
    pf_list = new_list

    if len(pf_list) > n_pf:
        raise ValueError('Invalid number of preprocessing functions')
    pf_list = pf_list + [fill_with_fnc for i in range(n_pf - len(pf_list))]
    return pf_list

def get_file_list(file_dir, file_ext, sub_name=None):
    """ Get file list in a directory with sepcific filename and extension

    Args:
        file_dir (str): directory of files
        file_ext (str): filename extension
        sub_name (str): Part of filename. Can be None.

    Return:
        List of filenames under ``file_dir`` as well as subdirectories

    """
    re_list = []

    if sub_name is None:
        return np.array([os.path.join(root, name)
            for root, dirs, files in os.walk(file_dir) 
            for name in sorted(files) if name.lower().endswith(file_ext)])
    else:
        return np.array([os.path.join(root, name)
            for root, dirs, files in os.walk(file_dir) 
            for name in sorted(files)
            if name.lower().endswith(file_ext) and sub_name.lower() in name.lower()])

_RNG_SEED = None

def get_rng(obj=None):
    """
    This function is copied from `tensorpack
    <https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/utils.py>`__.
    Get a good RNG seeded with time, pid and the object.

    Args:
        obj: some object to use to generate random seed.

    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return np.random.RandomState(seed)


