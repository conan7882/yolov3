#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: augmentation.py
# Author: Qian Ge <geqian1001@gmail.com>

import colorsys
import skimage.transform
import matplotlib.colors
import numpy as np
from numpy.linalg import inv
import src.utils.image as imagetool
import src.bbox.bboxtool as bboxtool

SMALL_VAL = 1e-6

def im_preserve_range(image, intensity_range):
    # intensity_range 1. or  255.
    return np.clip(image, 0., intensity_range)

def remove_invalid_bbox(image, bbox):
    bbox[np.where(bbox < 0)] = 0
    invalid_id = np.any([(bbox[..., 0] >= image.shape[1]), (bbox[..., 1] >= image.shape[0])], axis=0)
    bbox[invalid_id] = [0, 0, 0, 0]

    bbox[..., 2][np.where(bbox[..., 2] >= image.shape[1])] = image.shape[1] - 1
    bbox[..., 3][np.where(bbox[..., 3] >= image.shape[0])] = image.shape[0] - 1
    return bbox

def horizontal_flip(image, bboxes):
    """ horizontal flip image and bboxes

        Args:
            image (np.array): [h, w, c]
            bboxes: [n_bbox, 4] xyxy

        Returns:
            flipped image and bboxes
    """
    flip_image = np.fliplr(image)
    bboxes[..., [2, 0]] = image.shape[1] - bboxes[..., [0, 2]]
    return flip_image, bboxes

def crop(image, bboxes, crop_size):
    """ center crop image and bboxes
        cropped shape is min(im_shape, crop_shape)

        Args:
            image (np.array): [h, w, c]
            crop_size (list of length 4): cropped size [starty, startx, h, w] must be small than image size
            bboxes: [n_bbox, 4] xyxy

        Returns:
            cropped image and bboxes
    """
    s_y, s_x, h, w = crop_size
    cropped_im = image[s_y: s_y + h, s_x: s_x + w]
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] - s_x
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] - s_y
    return cropped_im, bboxes

def center_crop(image, bboxes, crop_shape):
    """ center crop image and bboxes
        cropped shape is min(im_shape, crop_shape)

        Args:
            image (np.array): [h, w, c]
            crop_shape (list of length 2): cropped size [h, w] must be less than image shape
            bboxes: [n_bbox, 4] xyxy

        Returns:
            cropped image and bboxes
    """
    im_shape = image.shape
    new_h, new_w = crop_shape[0], crop_shape[1]
    s_h, s_w = (im_shape[0] - new_h) // 2, (im_shape[1] - new_w) // 2
    cropped_im = image[s_h: s_h + new_h, s_w: s_w + new_w]
    
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] - s_w
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] - s_h
    return cropped_im, bboxes

def rescale(image, bboxes, rescale_shape):
    """ rescale image and bboxes

        Args:
            image (np.array): [h, w, c]
            rescale_shape (list of length 2): rescale size [h, w]
            bboxes: [n_bbox, 4] xyxy

        Returns:
            rescaled image and bboxes
    """

    rescale_im = skimage.transform.resize(
        image, rescale_shape, preserve_range=True, mode='reflect')
    from_shape = image.shape
    to_shape = rescale_shape

    
    bboxes = np.stack(
        [bboxes[..., 0] / from_shape[1] * to_shape[1],
         bboxes[..., 1] / from_shape[0] * to_shape[0],
         bboxes[..., 2] / from_shape[1] * to_shape[1],
         bboxes[..., 3] / from_shape[0] * to_shape[0]], axis=-1)

    # if np.any(bboxes > 320):
    #     print(bboxes, from_shape, to_shape)

    return rescale_im, bboxes

def affine_transform(image, bboxes, scale=[1., 1.], translation=[0., 0.], shear=[0., 0.], angle=0):
    """ rescale image and bboxes

        Args:
            image (np.array): [h, w, c]
            bboxes: [n_bbox, 4] xyxy
            angle (float): rotation angle in degrees in counter-clockwise direction [-180, 180]
            shear (list of 2)
            translation (list of 2): [-1, 1]

        Returns:
            rotated image and bboxes
    """
    def _gen_affine_trans():
        s_x = scale[0]
        s_y = scale[1]

        t_x = translation[0]
        t_y = translation[1]

        theta = np.pi * angle / 180
        sin = np.sin(theta)
        cos = np.cos(theta)

        matrix = [[s_x * cos,  -s_y * sin, t_x * s_x * cos - t_y * s_y * sin], 
                  [s_x * sin, s_y * cos, t_x * s_x * sin + t_y * s_y * cos,],
                  [0,0,1]]

        shear_x = shear[0]
        shear_y = shear[1]
        shear_matrix = [[1. + shear_x * shear_y,  shear_x, 0.], 
                        [shear_y, 1., 0],
                        [0., 0. , 1.]]

        return np.matmul(matrix, shear_matrix)

    T = _gen_affine_trans()  
    rotate_im = imagetool.affine_transform(image, T, out_shape=None)
    # TODO 
    # faster compute inv of matrix
    bboxes = bboxtool.affine_transform_bbox(bboxes, inv(T), image.shape, rotate_im.shape)

    return rotate_im, bboxes

def change_color(image, hue, saturate, brightness, intensity_scale=255.):
    """ 

        Args:
            image (np.array): color image with shape [h, w, c]
            hue (float): [0, 1]
            intensity_scale (float): 1. or 255. scale of image intensity

        Returns:
    """
    rgb_im = image * 1. / intensity_scale
    hsv_image = matplotlib.colors.rgb_to_hsv(rgb_im)

    rgb_to_hsv(rgb_im)

    hsv_image[..., 0] += hue
    hsv_image[..., 0][np.where(hsv_image[..., 0] > 1)] = hsv_image[..., 0][np.where(hsv_image[..., 0] > 1)] - 1
    hsv_image[..., 0][np.where(hsv_image[..., 0] < 0)] = hsv_image[..., 0][np.where(hsv_image[..., 0] < 0)] + 1
    change_list = [saturate, brightness]
    for i in range(2):
        hsv_image[..., i] *= change_list[i] 

    rgb_im = matplotlib.colors.hsv_to_rgb(hsv_image) * intensity_scale
    return rgb_im

def rgb_to_hsv(im):
    # im (np.array) [..., h, w, 3] rgb range [0, 1]

    im = np.reshape(im, (-1, im.shape[-2]*im.shape[-3], 3))
    r, g, b = np.split(im, 3, axis=-1)
    max_channel_id = np.argmax(im, axis=-1)
    # 

    c_max = np.expand_dims(np.amax(im, axis=-1), axis=-1)
    c_min = np.expand_dims(np.amin(im, axis=-1), axis=-1)

    delta = c_max - c_min
    print(max_channel_id.shape)
    H = np.stack([60 * ((g-b) / (delta + SMALL_VAL) % 6),
                  60 * ((b-r) / (delta + SMALL_VAL) + 2),
                  60 * ((r-g) / (delta + SMALL_VAL) + 4)], axis=-1)
    print(H.shape)
    print(H[..., max_channel_id[0][0]].shape)

    S = delta / (c_max + SMALL_VAL)
    # print(S.shape)

    V = c_max
    # print(V.shape)


