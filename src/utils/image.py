#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: image.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import skimage.transform

def rescale_image(image, rescale_shape):
    if isinstance(rescale_shape, int):
        rescale_shape = [rescale_shape, rescale_shape]
    rescale_im = skimage.transform.resize(
        image, rescale_shape, preserve_range=True, mode='reflect')
    return rescale_im

# def gen_affine_trans(bsize):
#     s_x = np.random.uniform(low=0.7, high=1.3, size=[bsize, 1])
#     s_y = np.random.uniform(low=0.7, high=1.3, size=[bsize, 1])

#     t_x = np.random.uniform(low=-0.5, high=0.5, size=[bsize, 1])
#     t_y = np.random.uniform(low=-0.5, high=0.5, size=[bsize, 1])

#     theta = np.pi * np.random.uniform(low=-0.25, high=0.25, size=[bsize, 1])

#     # s_x = s_y = np.ones((bsize, 1))
#     # t_x = t_y = np.zeros((bsize, 1))
#     # # 
#     # theta = np.pi * 0.5 * np.ones((bsize, 1))

#     matrix = np.concatenate(
#       (s_x * np.cos(theta), -s_y * np.sin(theta),
#       t_x * s_x * np.cos(theta) - t_y * s_y * np.sin(theta),
#       s_x * np.sin(theta), s_y * np.cos(theta),
#       t_x * s_x * np.sin(theta) + t_y * s_y * np.cos(theta),
#       np.zeros((bsize, 1)), np.zeros((bsize, 1)), np.ones((bsize, 1))),
#       axis=-1)

#     # s_x = 1 / s_x
#     # s_y = 1 / s_y
#     # t_x = -t_x
#     # t_y = -t_y
#     # theta = -theta

#     # inv_matrix = np.concatenate(
#     #   (s_x * np.cos(theta), -s_y * np.sin(theta),
#     #   t_x * s_x * np.cos(theta) - t_y * s_y * np.sin(theta),
#     #   s_x * np.sin(theta), s_y * np.cos(theta),
#     #   t_x * s_x * np.sin(theta) + t_y * s_y * np.cos(theta),
#     #   np.zeros((bsize, 1)), np.zeros((bsize, 1)), np.ones((bsize, 1))),
#     #   axis=-1)

#     matrix = np.reshape(matrix, (-1, 3, 3))
#     # inv_matrix = np.reshape(inv_matrix, (-1, 3, 3))
#     return matrix

def get_pixel_value(inputs, x, y):
    return inputs[y, x]


def affine_transform(inputs, T, out_shape=None):
    """
    Args:
        T [2, 3]
        outdim (list of 2)
    """
    bsize = np.shape(inputs)[0]
    channel = np.shape(inputs)[-1]
    if out_shape is not None:
        h = out_shape[0]
        w = out_shape[1]
        # w = out_dim[1]
    else:
        h = np.shape(inputs)[0]
        w = np.shape(inputs)[1]

    o_h = np.shape(inputs)[0]
    o_w = np.shape(inputs)[1]

    x = np.linspace(-1., 1., w)
    y = np.linspace(-1., 1., h)
    X, Y = np.meshgrid(x, y)
    X_flatten = np.reshape(X, (-1,))
    Y_flatten = np.reshape(Y, (-1,))
    ones = np.ones_like(Y_flatten)
    homogeneous_coord = np.stack((X_flatten, Y_flatten, ones), axis=0) # [3, N]
    # homogeneous_coord = np.expand_dims(homogeneous_coord, axis=0) # [1, 3, N]
    # homogeneous_coord = np.tile(homogeneous_coord, (bsize, 1, 1)) # [bsize, 3, N]
    affine_transform_coord = np.matmul(T, homogeneous_coord) # [bsize, 2, N]
    affine_x = affine_transform_coord[..., 0, :]
    affine_y = affine_transform_coord[..., 1, :]
    # back to original scale
    affine_x = (affine_x + 1.) / 2. * float(o_w)
    affine_y = (affine_y + 1.) / 2. * float(o_h)
    # find four cornors
    x0 = (np.floor(affine_x)).astype(np.int32)
    x1 = x0 + 1
    y0 = (np.floor(affine_y)).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, o_w - 1)
    x1 = np.clip(x1, 0, o_w - 1)
    y0 = np.clip(y0, 0, o_h - 1)
    y1 = np.clip(y1, 0, o_h - 1)

    # # get value
    I00 = get_pixel_value(inputs, x0, y0)
    I10 = get_pixel_value(inputs, x1, y0)
    I01 = get_pixel_value(inputs, x0, y1)
    I11 = get_pixel_value(inputs, x1, y1)

    x0 = x0.astype(np.float32)
    x1 = x1.astype(np.float32)
    y0 = y0.astype(np.float32)
    y1 = y1.astype(np.float32)
    # compute weight
    w00 = (y1 - affine_y) * (x1 - affine_x)
    w10 = (y1 - affine_y) * (affine_x - x0)
    w01 = (affine_y - y0) * (x1 - affine_x)
    w11 = (affine_y - y0) * (affine_x - x0)

    w00 = np.expand_dims(w00, axis=-1)
    w10 = np.expand_dims(w10, axis=-1)
    w01 = np.expand_dims(w01, axis=-1)
    w11 = np.expand_dims(w11, axis=-1)

    transform_im = I00 * w00 + I10 * w10 + I01 * w01 + I11 * w11
    transform_im = np.reshape(transform_im, (h, w, channel))

    return transform_im

