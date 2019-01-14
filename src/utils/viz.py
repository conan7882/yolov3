#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz.py
# Author: Qian Ge <geqian1001@gmail.com>

import imageio
import numpy as np
import tensorflow as tf
import matplotlib
import platform
import scipy.misc
if platform.node() == 'arostitan':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import src.bbox.tfbboxtool as tfbboxtool
import src.bbox.bboxtool as bboxtool
from src.utils.visualization_utils import draw_bounding_boxes_on_image_tensors
from src.utils.visualization_utils import visualize_boxes_and_labels_on_image_array

def draw_bounding_box_on_image_array(im, bbox_list, class_list, score_list,
                                     category_index, save_name=None, save_fig=False):
    im = np.array(im).astype(np.uint8)
    bbox_im = im
    if len(class_list) > 0:
        viz_bbox = bboxtool.xyxy2yxyx(bbox_list)
        class_list = list(map(int, class_list))

        bbox_im = visualize_boxes_and_labels_on_image_array(
            image=im,
            boxes=viz_bbox,
            classes=class_list,
            scores=score_list,
            category_index=category_index,
            instance_masks=None,
            keypoints=None,
            use_normalized_coordinates=False,
            max_boxes_to_draw=20,
            min_score_thresh=.1,
            agnostic_mode=False,
            line_thickness=4)
    

    if save_fig and save_name is not None:
        scipy.misc.imsave(save_name, bbox_im)
        # plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    else:
        plt.figure()
        plt.imshow(bbox_im)
        plt.axis('off')
        plt.show()

    plt.close()

def tf_draw_bounding_box(im, bbox_list, score_list, class_list, category_index,
                         max_boxes_to_draw=20, min_score_thresh=0.5, box_type='xyxy'):

    im = tf.cast(im, tf.uint8)
    bbox_list = tf.cast(bbox_list, tf.float32)

    if box_type == 'xyxy':
        pass
    elif box_type == 'cxywh':
        bbox_list = tfbboxtool.cxywh2xyxy(bbox_list)
    else:
        raise ValueError('Incorrect box_type {}'.format(box_type))

    # viz_bbox = tfbboxtool.xyxy2yxyx(bbox_list)
    class_list = tf.cast(class_list, tf.int32)

    # change box to [y_min, x_min, y_max, x_max]
    im_h = tf.cast(tf.shape(im)[1], tf.float32)
    im_w = tf.cast(tf.shape(im)[2], tf.float32)

    bbox_list = tf.stack([bbox_list[..., 1] / im_h, bbox_list[..., 0] / im_w,
                          bbox_list[..., 3] / im_h, bbox_list[..., 2] / im_w], axis=-1)
    # bbox_list = tf.expand_dims(bbox_list, aixs=0)

    bbox_im = draw_bounding_boxes_on_image_tensors(
            im,
            bbox_list,
            class_list,
            score_list,
            category_index,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_score_thresh)

    return bbox_im

def draw_bounding_box(im, box, label_list=None, box_type='xyxy'):
    im = np.array(im, dtype=np.uint8)
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    if len(box) > 0:
        box = np.array(box)
        if len(box.shape) == 1:
            box = [box]
        # Create a Rectangle patch
        for idx, c_box in enumerate(box):
            if box_type == 'xyxy':
                rect = patches.Rectangle(
                    (c_box[0], c_box[1]), c_box[2] - c_box[0], c_box[3] - c_box[1],
                    linewidth=2, edgecolor='r', facecolor='none')
                x, y = c_box[0], c_box[1]
            elif box_type == 'cxywh':
                rect = patches.Rectangle(
                    (c_box[0] - c_box[2] / 2, c_box[1]- c_box[3] / 2), c_box[2], c_box[3],
                    linewidth=2, edgecolor='r', facecolor='none')
                x, y = c_box[0] - c_box[2] / 2, c_box[1]- c_box[3] / 2

            # Add the patch to the Axes
            ax.add_patch(rect)
            if label_list is not None:
                ax.text((x)/im.shape[0], 1 - (y)/im.shape[1], s=label_list[idx],
                        fontdict={'color': 'darkred', 'weight': 'bold'},
                        horizontalalignment='center',verticalalignment='center',
                        transform=ax.transAxes)
    ax.axis('off')
    plt.show()

def viz_batch_im(batch_im, grid_size, save_path,
                 gap=0, gap_color=0, shuffle=False):
    """ save batch of image as a single image 

    Args:
        batch_im (list): list of images 
        grid_size (list of 2): size (number of samples in row and column) of saving image
        save_path (str): directory for saving sampled images
        gap (int): number of pixels between two images
        gap_color (int): color of gap between images
        shuffle (bool): shuffle batch images for saving or not
    """

    batch_im = np.array(batch_im)
    if len(batch_im.shape) == 4:
        n_channel = batch_im.shape[-1]
    elif len(batch_im.shape) == 3:
        n_channel = 1
        batch_im = np.expand_dims(batch_im, axis=-1)
    assert len(grid_size) == 2

    h = batch_im.shape[1]
    w = batch_im.shape[2]

    merge_im = np.zeros((h * grid_size[0] + (grid_size[0] + 1) * gap,
                         w * grid_size[1] + (grid_size[1] + 1) * gap,
                         n_channel)) + gap_color

    n_viz_im = min(batch_im.shape[0], grid_size[0] * grid_size[1])
    if shuffle == True:
        pick_id = np.random.permutation(batch_im.shape[0])
    else:
        pick_id = range(0, batch_im.shape[0])
    for idx in range(0, n_viz_im):
        i = idx % grid_size[1]
        j = idx // grid_size[1]
        cur_im = batch_im[pick_id[idx], :, :, :]
        merge_im[j * (h + gap) + gap: j * (h + gap) + h + gap,
                 i * (w + gap) + gap: i * (w + gap) + w + gap, :]\
            = (cur_im)

    imageio.imwrite(save_path, np.squeeze(merge_im))

def display(global_step,
            step,
            scaler_sum_list,
            name_list,
            collection,
            summary_val=None,
            summary_writer=None,
            ):
    """ Display averaged intermediate results for a period during training.

    The intermediate result will be displayed as:
    [step: global_step] name_list[0]: scaler_sum_list[0]/step ...
    Those result will be saved as summary as well.

    Args:
        global_step (int): index of current iteration
        step (int): number of steps for this period
        scaler_sum_list (float): list of summation of the intermediate
            results for this period
        name_list (str): list of display name for each intermediate result
        collection (str): list of graph collections keys for summary
        summary_val : additional summary to be saved
        summary_writer (tf.FileWriter): write for summary. No summary will be
            saved if None.
    """
    print('[step: {}]'.format(global_step), end='')
    for val, name in zip(scaler_sum_list, name_list):
        print(' {}: {:.4f}'.format(name, val * 1. / step), end='')
    print('')
    if summary_writer is not None:
        s = tf.Summary()
        for val, name in zip(scaler_sum_list, name_list):
            s.value.add(tag='{}/{}'.format(collection, name),
                        simple_value=val * 1. / step)
        summary_writer.add_summary(s, global_step)
        if summary_val is not None:
            summary_writer.add_summary(summary_val, global_step)
