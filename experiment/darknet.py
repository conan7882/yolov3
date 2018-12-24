#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: darknet.py
# Author: Qian Ge <geqian1001@gmail.com>

# import os
import argparse
import platform
# import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
import loader
from src.net.darknet import DarkNet53


PRETRINED_PATH = '/Users/gq/workspace/Dataset/pretrained/darknet53_448.npy'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_path', type=str, default=PRETRINED_PATH,
                        help='Directory of pretrain model')
    parser.add_argument('--im_name', type=str, default='.jpg',
                        help='Part of image name')
    parser.add_argument('--data_dir', type=str, default='../data/',
                        help='Directory of test images')

    parser.add_argument('--rescale', type=int, default=256,
                        help='Rescale input image with shorter side = rescale')
    
    return parser.parse_args()

def predict():
    FLAGS = get_args()
    label_dict = loader.load_imagenet1k_label_darknet()
    # Create a Dataflow object for test images
    image_data = loader.read_image(
        im_name=FLAGS.im_name, n_channel=3,
        data_dir=FLAGS.data_dir, batch_size=1, rescale=FLAGS.rescale)

    test_model = DarkNet53(
        n_channel=3, n_class=1000, pre_trained_path=FLAGS.pretrained_path, trainable=False)
    test_model.create_valid_model()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while image_data.epochs_completed < 1:
            batch_data = image_data.next_batch_dict()
            # get batch file names
            batch_file_name = image_data.get_batch_file_name()[0]
            # get prediction results
            pred = sess.run(test_model.layers['top_5'],
                            feed_dict={test_model.image: batch_data['image']})
            # display results
            for re_prob, re_label, file_name in zip(pred[0], pred[1], batch_file_name):
                print('===============================')
                print('[image]: {}'.format(file_name))
                for i in range(5):
                    print('{}: probability: {:.02f}, label: {}'
                          .format(i+1, re_prob[i], label_dict[re_label[i]]))


if __name__ == '__main__':
    predict()