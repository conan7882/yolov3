#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: generator.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from src.dataflow.preprocess import PreProcess


class Generator(object):
    def __init__(self, dataflow, rescale_shape_list, stride_list, prior_list, n_class,
                 batch_size, buffer_size, num_parallel_preprocess,
                 h_flip=False, crop=False, color=False, affine=False, im_intensity = 1.,
                 max_num_bbox_per_im=45):

        self._im_scale_list = rescale_shape_list
        dataflow.set_batch_size(1)

        self.preprocessor = PreProcess(
            dataflow=dataflow, 
            rescale_shape_list=rescale_shape_list,
            stride_list=stride_list, 
            prior_list=prior_list, 
            n_class=n_class,
            h_flip=h_flip, 
            crop=crop, 
            color=color, 
            affine=affine,
            max_num_bbox_per_im=max_num_bbox_per_im)

        self.reset_im_scale(scale=None)

        def generator():
            dataflow.reset_epochs_completed()
            cnt = 0
            while dataflow.epochs_completed < 1:
                cnt += 1
                print(cnt)
                batch_data = dataflow.next_batch_dict()
                yield batch_data['image'], batch_data['label']

        dataset = tf.data.Dataset().from_generator(
            generator, output_types= (tf.float32, tf.float32),)

        dataset = dataset.map(map_func=self.preprocessor.tf_process_batch,
                              num_parallel_calls=num_parallel_preprocess)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)

        self.iter = dataset.make_initializable_iterator()
        self.batch_data = self.iter.get_next()

    def init_iterator(self, sess):
        sess.run(self.iter.initializer)

    def reset_im_scale(self, scale=None):
        if scale is not None:
            self.preprocessor.set_output_scale(scale)
        else:
            pick_id = np.random.choice(len(self._im_scale_list))
            scale = self._im_scale_list[pick_id]
            self.preprocessor.set_output_scale(scale)

