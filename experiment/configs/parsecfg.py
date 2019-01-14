#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: parsecfg.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import configparser
import numpy as np


class ConfigParser(object):
    def __init__(self, path_cfg_path, net_cfg_path):
        self.pathconfig = self.parse_cfg(path_cfg_path)
        self.netconfig = self.parse_cfg(net_cfg_path)
        self._get_config()

    def _get_value(self, parser, name):
        try:
            return parser[name]
        except KeyError:
            return None

    def _get_config(self):
        self.coco_pretrained_path = self._get_value(self.pathconfig, 'coco_pretrained_npy_path')
        self.yolo_feat_pretrained_path = self._get_value(self.pathconfig, 'yolo_feat_pretraind_npy')
        
        self.data_dir = self._get_value(self.pathconfig, 'test_image_path')
        self.save_path = self._get_value(self.pathconfig, 'save_path')
        self.im_name = self._get_value(self.pathconfig, 'test_image_name')

        self.train_data_dir = self._get_value(self.pathconfig, 'train_data_path')

        self.im_rescale = self._get_value(self.netconfig, 'rescale')
        self.mutliscale = self._get_value(self.netconfig, 'multiscale')
        self.n_channel = self._get_value(self.netconfig, 'n_channel')
        self.test_bsize = self._get_value(self.netconfig, 'test_bsize')
        self.train_bsize = self._get_value(self.netconfig, 'train_bsize')
        self.obj_score_thr = self._get_value(self.netconfig, 'obj_score_thr')
        self.nms_iou_thr = self._get_value(self.netconfig, 'nms_iou_thr')
        self.n_class = self._get_value(self.netconfig, 'n_class')
        self.anchors = self._get_value(self.netconfig, 'anchors')
        self.ignore_thr = self._get_value(self.netconfig, 'ignore_thr')
        self.obj_weight = self._get_value(self.netconfig, 'obj_weight')
        self.nobj_weight = self._get_value(self.netconfig, 'nobj_weight')

    def _get_scalar_value(self, parser, val_type, section, name):
        try:
            return val_type(parser[section][name])
        except KeyError:
            return None

    def _get_list(self, parser, val_type, section, name):
        try:
            return list(map(val_type, (parser[section][name]).split(',')))
        except KeyError:
            return None

    def _get_str(self, parser, section, name):
        try:
            return parser[section][name]
        except KeyError:
            return None

    def parse_cfg(self, file_name):
        # section_counters = defaultdict(int)
        cfg_stream = open(file_name)
        cfg_parser = configparser.ConfigParser()
        cfg_parser.read_file(cfg_stream)

        cfg_dict = {}
        for section in cfg_parser.sections():
            if section.startswith('net'):
                cfg_dict['test_bsize'] = self._get_scalar_value(cfg_parser, int, section, 'test_batch')
                cfg_dict['train_bsize'] = self._get_scalar_value(cfg_parser, int, section, 'train_batch')
                cfg_dict['rescale'] = [self._get_scalar_value(cfg_parser, int, section, 'width'),
                                       self._get_scalar_value(cfg_parser, int, section, 'height')]
                cfg_dict['n_channel'] = self._get_scalar_value(cfg_parser, int, section, 'channel')
                cfg_dict['multiscale'] = self._get_list(cfg_parser, int, section, 'multiscale')

            if section.startswith('yolo'):
                cfg_dict['obj_score_thr'] = self._get_scalar_value(cfg_parser, float, section, 'obj_score_thresh')
                cfg_dict['nms_iou_thr'] = self._get_scalar_value(cfg_parser, float, section, 'nms_iou_thresh')
                cfg_dict['n_class'] = self._get_scalar_value(cfg_parser, int, section, 'classes')
                cfg_dict['ignore_thr'] = self._get_scalar_value(cfg_parser, float, section, 'ignore_thr')
                cfg_dict['nobj_weight'] = self._get_scalar_value(cfg_parser, float, section, 'nobj_weight')
                cfg_dict['obj_weight'] = self._get_scalar_value(cfg_parser, float, section, 'obj_weight')

                anchors = self._get_list(cfg_parser, float, section, 'anchors')
                anchor_mask = self._get_list(cfg_parser, int, section, 'anchor_mask')

                anchor_scale = np.amax(anchor_mask)
                anchor_list = [[] for _ in range(anchor_scale+1)]
                for idx, a_ind in enumerate(anchor_mask):
                    anchor_list[a_ind].append(anchors[idx*2: idx*2+2])
                cfg_dict['anchors'] = anchor_list

            if section.startswith('path'):
                cfg_dict['coco_pretrained_npy_path'] = self._get_str(cfg_parser, section, 'coco_pretrained_npy')
                # cfg_parser[section]['coco_pretrained_npy']
                cfg_dict['yolo_feat_pretraind_npy'] = self._get_str(cfg_parser, section, 'yolo_feat_pretraind_npy')
                # cfg_parser[section]['yolo_feat_pretraind_npy']
                cfg_dict['save_path'] = self._get_str(cfg_parser, section, 'save_path')
                # cfg_parser[section]['save_path']
                
                cfg_dict['test_image_path'] = self._get_str(cfg_parser, section, 'test_image_path')
                cfg_dict['train_data_path'] = self._get_str(cfg_parser, section, 'train_data_path')
                # cfg_parser[section]['test_image']
                cfg_dict['test_image_name'] = self._get_str(cfg_parser, section, 'test_image_name')
                # cfg_parser[section]['test_image_name']

        return cfg_dict

if __name__ == '__main__':
    cfg_dict = parse_cfg('path.cfg')
    print(cfg_dict)
