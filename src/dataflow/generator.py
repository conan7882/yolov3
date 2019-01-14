#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: generator.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from src.dataflow.preprocess import PreProcess


class Generator(object):
    def __init__(self, dataflow, n_channle, rescale_shape_list, stride_list, prior_list, n_class,
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
        self.output_scale = self.preprocessor.output_scale
        self._n_preprocess = num_parallel_preprocess

        # if not isinstance(dataflow_key_list, list):
        #     dataflow_key_list = [dataflow_key_list]

        def generator():
            dataflow.reset_epochs_completed()
            while dataflow.epochs_completed < 1:
                batch_data = dataflow.next_batch_dict()
                yield batch_data['image'], batch_data['label']

        dataset = tf.data.Dataset().from_generator(
            generator,
            output_types= (tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([1, None, None, n_channle]), tf.TensorShape([1, None, 5])),
            )

        # dataset = dataset.map(lambda x, y: self.preprocessor.tf_process_batch(x, y, 2),
        #                       num_parallel_calls=self._n_preprocess)
        dataset = dataset.map(map_func=self.preprocessor.tf_process_batch,
                              num_parallel_calls=num_parallel_preprocess)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)

        self.iter = dataset.make_initializable_iterator()
        self.batch_data = self.iter.get_next()

        self.dataset = dataset

    def init_iterator(self, sess, reset_scale=False):
        if reset_scale:
            self.reset_im_scale()
        sess.run(self.iter.initializer, feed_dict={self.output_scale: self._image_scale})

    def reset_im_scale(self, scale=None):
        if scale is not None:
            self._image_scale = scale
            # self.preprocessor.set_output_scale(scale)
        else:
            pick_id = np.random.choice(len(self._im_scale_list))
            scale = self._im_scale_list[pick_id]
            self._image_scale = scale
            # self.preprocessor.set_output_scale(scale)

        print('rescale to {}'.format(scale))

