#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: image.py
# Author: Qian Ge <geqian1001@gmail.com>

import skimage.transform

def rescale_image(image, rescale_shape):
    if isinstance(rescale_shape, int):
        rescale_shape = [rescale_shape, rescale_shape]
    rescale_im = skimage.transform.resize(
        image, rescale_shape, preserve_range=True, mode='reflect')
    return rescale_im