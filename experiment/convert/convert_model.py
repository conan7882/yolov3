#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: convert_model.py
# Author: Qian Ge <geqian1001@gmail.com>
# Modified from 
# https://github.com/qqwweee/keras-yolo3/blob/master/convert.py
# reference:
# https://github.com/pjreddie/darknet/blob/b13f67bfdd87434e141af532cdb5dc1b8369aa3b/src/parser.c#L958

import io
import os
import argparse
import configparser
import platform
from collections import defaultdict
import numpy as np


def unique_config_sections(config_path):
    """Convert all config sections to have unique names.
    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    yolo_id = 0
    # prev_dim = 3
    # prev_dim_dict = {}
    with open(config_path) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                n_section = section
                # out_dim = prev_dim
                if section == 'yolo':
                    n_section = section
                    yolo_id += 1
                elif yolo_id < 1:
                    if section == 'convolutional':
                        n_section = 'conv'
                        if section_counters[n_section] == 51:
                            yolo_id = 1
                    elif section == 'convolutional_fc':
                        n_section = 'conv_fc'
                elif yolo_id > 0:
                    if section == 'convolutional':
                        n_section = 'conv_{}'.format(yolo_id)                    

                section_counters[n_section] += 1
                if yolo_id > 1:
                    _section = n_section + '_' + str(section_counters[n_section]-1)
                else:
                    _section = n_section + '_' + str(section_counters[n_section])
                # prev_dim_dict[_section] = prev_dim
                # prev_dim = out_dim
                print(_section)
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream

def parse_conv(weights_file, cfg_parser, section, layer_dict):
    prev_layer_channel = layer_dict['prev_layer_channel']
    count = layer_dict['count']

    filters = int(cfg_parser[section]['filters'])
    size = int(cfg_parser[section]['size'])
    stride = int(cfg_parser[section]['stride'])
    pad = int(cfg_parser[section]['pad'])
    activation = cfg_parser[section]['activation']
    batch_normalize = 'batch_normalize' in cfg_parser[section]

    # Setting weights.
    # Darknet serializes convolutional weights as:
    # [bias/beta, [gamma, mean, variance], conv_weights]
    # prev_layer_shape = K.int_shape(prev_layer)

    weights_shape = (size, size, prev_layer_channel, filters)
    darknet_w_shape = (filters, weights_shape[2], size, size)
    weights_size = np.product(weights_shape)
    prev_layer_channel = filters

    print('conv2d', 'bn'
          if batch_normalize else '  ', activation, weights_shape)

    bn_weight_list = []
    conv_bias = []
    if batch_normalize:
        bn_weights = np.ndarray(
            shape=(4, filters),
            dtype='float32',
            buffer=weights_file.read(filters * 16))
        count += 4 * filters

        bn_weight_list = [
            bn_weights[1],  # scale gamma
            bn_weights[0],  # shift beta
            bn_weights[2],  # running mean
            bn_weights[3]  # running var
        ]
    else:
        conv_bias = np.ndarray(
        shape=(filters, ),
        dtype='float32',
        buffer=weights_file.read(filters * 4))
        count += filters

    conv_weights = np.ndarray(
        shape=darknet_w_shape,
        dtype='float32',
        buffer=weights_file.read(weights_size * 4))
    count += weights_size

    # DarkNet conv_weights are serialized Caffe-style:
    # (out_dim, in_dim, height, width)
    # We would like to set these to Tensorflow order:
    # (height, width, in_dim, out_dim)
    conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])

    layer_dict['prev_layer_channel'] = prev_layer_channel
    layer_dict['count'] = count
    layer_dict['conv_weights'] = conv_weights
    layer_dict['conv_bias'] = conv_bias
    layer_dict['bn_weight_list'] = bn_weight_list

    return layer_dict

def convert(weights_path, config_path, save_path):
    weights_file = open(weights_path, 'rb')
    major, minor, revision = np.ndarray(
        shape=(3, ), dtype='int32', buffer=weights_file.read(12))

    if (major*10+minor)>=2 and major<1000 and minor<1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)

    net_config = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(net_config)

    save_weight_dict = {}
    layer_dict = {}
    dim_list = []
    layer_dict['prev_layer_channel'] = 3
    layer_dict['count'] = 0
    # layer_id = 0
    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))
        if section.startswith('conv'):
            save_weight_dict[section] = {}

            layer_dict = parse_conv(weights_file, cfg_parser, section, layer_dict)
            save_weight_dict[section]['weights'] = layer_dict['conv_weights']
            
            if len(layer_dict['bn_weight_list']) > 0:
               save_weight_dict[section]['bn'] = layer_dict['bn_weight_list']
            if len(layer_dict['conv_bias']) > 0:
                save_weight_dict[section]['biases'] = layer_dict['conv_bias']
        elif section.startswith('route'):
            route_layers = list(map(int, (cfg_parser[section]['layers']).split(',')))
            layer_dict['prev_layer_channel'] = sum([dim_list[layer_] for layer_ in route_layers])
            # print(route_layers)
        dim_list.append(layer_dict['prev_layer_channel'])
        # layer_id += 1
    remaining_weights = len(weights_file.read()) / 4
    print('Load {} of {} from weights.'.format(layer_dict['count'], remaining_weights + layer_dict['count']))
    weights_file.close()
    # print(dim_list)
    np.save(save_path, save_weight_dict)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights_dir', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')

    parser.add_argument('--model', type=str, default='yolo')
    
    return parser.parse_args()


if __name__ == '__main__':
    if platform.node() == 'arostitan':
        weights_dir = '/home/qge2/workspace/data/pretrain/yolo/'
    elif platform.node() == 'aros04':
        weights_dir = 'E:/Dataset/pretrained/'
    elif platform.node() == 'Qians-MacBook-Pro.local':
        weights_dir = '/Users/gq/workspace/Dataset/pretrained/'
        save_dir = '/Users/gq/workspace/Dataset/pretrained/'
    else:
        weights_dir = FLAGS.weights_dir
        save_dir = FLAGS.save_dir


    FLAGS = get_args()
    if FLAGS.model == 'darknet':
        config_path = 'darknet53.cfg'
        weights_path = os.path.join(weights_dir, 'darknet53_448.weights')
        save_path = os.path.join(save_dir, 'darknet53_448.npy')
    elif FLAGS.model == 'yolov3_feat':
        config_path = 'yolov3_feat.cfg'
        weights_path = os.path.join(weights_dir, 'yolov3.weights')
        save_path = os.path.join(save_dir, 'yolov3_feat.npy')
    elif FLAGS.model == 'yolo':
        config_path = 'yolov3.cfg'
        weights_path = os.path.join(weights_dir, 'yolov3.weights')
        save_path = os.path.join(save_dir, 'yolov3.npy')
    convert(weights_path, config_path, save_path)
    # unique_config_sections(config_path)
    
