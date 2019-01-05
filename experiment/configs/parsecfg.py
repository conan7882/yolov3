#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: parsecfg.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import configparser
import numpy as np


def Config(object):
    def __init__(self):
        pass



def parse_cfg(file_name):
    # section_counters = defaultdict(int)
    cfg_stream = open(file_name)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(cfg_stream)

    cfg_dict = {}
    for section in cfg_parser.sections():
        if section.startswith('net'):
            cfg_dict['test_bsize'] = int(cfg_parser[section]['test_batch'])
            cfg_dict['train_bsize'] = int(cfg_parser[section]['train_batch'])
            cfg_dict['rescale'] = [int(cfg_parser[section]['width']),
                                   int(cfg_parser[section]['height'])]
            cfg_dict['n_channel'] = int(cfg_parser[section]['channel'])
            cfg_dict['multiscale'] = list(map(int, (cfg_parser[section]['multiscale']).split(',')))

        if section.startswith('yolo'):
            cfg_dict['obj_score_thr'] = float(cfg_parser[section]['obj_score_thresh'])
            cfg_dict['nms_iou_thr'] = float(cfg_parser[section]['nms_iou_thresh'])
            cfg_dict['n_class'] = int(cfg_parser[section]['classes'])
            cfg_dict['ignore_thr'] = float(cfg_parser[section]['ignore_thr'])

            anchors = list(map(float, (cfg_parser[section]['anchors']).split(',')))
            anchor_mask = list(map(int, (cfg_parser[section]['anchor_mask']).split(',')))

            anchor_scale = np.amax(anchor_mask)
            anchor_list = [[] for _ in range(anchor_scale+1)]
            for idx, a_ind in enumerate(anchor_mask):
                anchor_list[a_ind].append(anchors[idx*2: idx*2+2])
            cfg_dict['anchors'] = anchor_list

        if section.startswith('path'):
            cfg_dict['coco_pretrained_npy_path'] = cfg_parser[section]['coco_pretrained_npy']
            cfg_dict['yolo_feat_pretraind_npy'] = cfg_parser[section]['yolo_feat_pretraind_npy']
            cfg_dict['save_path'] = cfg_parser[section]['save_path']
            
            cfg_dict['test_image_path'] = cfg_parser[section]['test_image']
            cfg_dict['test_image_name'] = cfg_parser[section]['test_image_name']

    return cfg_dict

if __name__ == '__main__':
    cfg_dict = parse_cfg('path.cfg')
    print(cfg_dict)
