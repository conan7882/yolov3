#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: voc.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import re
import scipy.misc
import numpy as np
import xml.etree.ElementTree as ET

from src.dataflow.base import DataFlow
import src.utils.utils as utils
# import src.bbox.bboxgt as bboxgt
from src.utils.dataflow import load_image, identity, fill_pf_list, get_file_list


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
        # try:
        #     self._class_dict[name]
        #     cur_class = self._class_dict[name]
        # except KeyError:
        #     self._class_dict[name] = self._nclass
        #     cur_class = self._class_dict[name]
        #     self._nclass += 1

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


class VOC(DataFlow):
    """ dataflow for CelebA dataset """
    def __init__(self,
                 class_dict,
                 image_dir='',
                 xml_dir='',
                 n_channel=3,
                 shuffle=True,
                 batch_dict_name=None,
                 pf_list=None):
        """
        Args:
            data_dir (str): directory of data
            shuffle (bool): whether shuffle data or not
            batch_dict_name (str): key of face image when getting batch data
            pf_list: pre-process functions for face image
        """

        pf_list = fill_pf_list(
            pf_list, n_pf=2, fill_with_fnc=identity)

        self._class_dict = class_dict
        self._nclass = len(class_dict)
        self._n_channel = n_channel
        self._pf_list = pf_list

        def read_image(file_name):
            """ read color face image with pre-process function """
            image = load_image(file_name, read_channel=n_channel,  pf=pf_list[0])
            return image

        def read_xml(xml_path, pf=pf_list[1]):
            """
                Returns:
                    [(class_id, [xmin, ymin, xmax, ymax])]
            """
            # [class_name, [xmin, ymin, xmax, ymax]]
            re = parse_bbox_xml(xml_path, self._class_dict)
            return re

        self.image_shape_dict = {}
        def read_shape(file_name):
            # be careful when the training set is too large
            try:
                return self.image_shape_dict[file_name]
            except KeyError:
                im = load_image(file_name, read_channel=n_channel)
                self.image_shape_dict[file_name] = im.shape[0:2]
                return self.image_shape_dict[file_name]

        super(VOC, self).__init__(
            data_name_list=['.jpg', '.xml', '.jpg'],
            data_dir=[image_dir, xml_dir, image_dir],
            shuffle=shuffle,
            batch_dict_name=batch_dict_name,
            load_fnc_list=[read_image, read_xml, read_shape],
            ) 

    def _load_file_list(self, data_name_list, data_dir_list):
        self._file_name_list = [[] for _ in range(len(data_name_list))]
        self._file_name_list[0] = get_file_list(data_dir_list[0], data_name_list[0])

        for file_path in self._file_name_list[0]:
            drive, path_and_file = os.path.splitdrive(file_path)
            path, file = os.path.split(path_and_file)
            file_id = re.findall(r'\d+', file)[0]

            for idx, (data_name, data_dir) in enumerate(zip(data_name_list[1:], data_dir_list[1:])):
                self._file_name_list[idx + 1].append(os.path.join(data_dir, file_id + data_name))

        for idx, file_list in enumerate(self._file_name_list):
            self._file_name_list[idx] = np.array(file_list)

        if self._shuffle:
            self._suffle_file_list()

    def reset_image_rescale(self, rescale):

        def read_image(file_name):
            """ read color face image with pre-process function """
            image = load_image(file_name, read_channel=self._n_channel,  pf=(self._pf_list[0][0], rescale))
            return image

        self._load_fnc_list[0] = read_image
