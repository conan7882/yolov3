#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: images.py
# Author: Qian Ge <geqian1001@gmail.com>

from src.dataflow.base import DataFlow
from src.utils.dataflow import load_image, identity, fill_pf_list


class Image(DataFlow):
    def __init__(self,
                 im_name,
                 data_dir='',
                 n_channel=3,
                 shuffle=True,
                 batch_dict_name=None,
                 pf_list=None):
        pf_list = fill_pf_list(
            pf_list, n_pf=1, fill_with_fnc=identity)

        def read_image(file_name):
            im = load_image(file_name, read_channel=n_channel,  pf=pf_list[0])
            return im

        # def read_shape(file_name):
        #     im = load_image(file_name, read_channel=n_channel)
        #     return im.shape[0:2]
        self.image_shape_dict = {}
        def read_shape(file_name):
            # be careful when the training set is too large
            try:
                return self.image_shape_dict[file_name]
            except KeyError:
                im = load_image(file_name, read_channel=n_channel)
                self.image_shape_dict[file_name] = im.shape[0:2]
                return self.image_shape_dict[file_name]

        super(Image, self).__init__(
            data_name_list=[im_name, im_name],
            data_dir=[data_dir, data_dir],
            shuffle=shuffle,
            batch_dict_name=batch_dict_name,
            load_fnc_list=[read_image, read_shape],
            ) 