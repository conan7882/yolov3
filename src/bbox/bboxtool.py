#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bboxtool.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np

SMALL_NUM = 1e-6

# def correct_yolo_boxes(xy_grid_flatten, bbox, anchor, scale):
#     # [bsize, h, w, 4]
#     bsize = tf.shape(bbox)[0]
#     shape = tf.shape(bbox)
#     bbox_flatten = tf.reshape(bbox, (bsize, -1, 4))
#     bbox_xy, bbox_wh = tf.split(bbox_flatten, [2, 2], axis=-1)
#     bbox_xy = tf.nn.sigmoid(bbox_xy)
#     bbox_xy = bbox_xy + xy_grid_flatten

#     pw, ph = anchor[0], anchor[1]
#     bwh = tf.multiply(anchor, tf.exp(bbox_wh))

#     correct_bbox = tf.concat([bbox_xy * scale, bwh], axis=-1)
#     return tf.reshape(correct_bbox, (bsize, shape[1], shape[2], 4))

def inverse_sigmoid(x):
    return np.log(x / (1 - x + SMALL_NUM) + SMALL_NUM)

def xyxy2yolotcoord(xyxy_bbox, anchor, stride, cxy):
    """ Map xyxy bbox to yolo prediction coordinates

        Args:
            xyxy_bbox (list): List of bbox with shape [n_bbox, 4].
                Each bbox is represented as [xmin, ymin, xmax, ymax]
            anchor (list): List of length 2 [anchor_w, anchor_h].
            stride (float): stride of current anchor 
            cxy (list): List of length 2. Center coordinates [x, y] of 
                this bbox on dowmsampled feature map. 
    """
    cxywh_bbox = xyxy2cxywh(np.array(xyxy_bbox))
    bbox_cxy, bbox_wh = np.split(cxywh_bbox, indices_or_sections=2, axis=-1)
    bbox_cxy = bbox_cxy / stride
    bcxy = inverse_sigmoid(bbox_cxy - cxy)
    bwh = np.log(bbox_wh / anchor)

    return np.concatenate([bcxy, bwh], axis=-1) 

def xyxy2cxywh(bbox_list):
    cxywh_bbox = np.stack(
        [(bbox_list[:, 0] + bbox_list[:, 2]) / 2,
         (bbox_list[:, 1] + bbox_list[:, 3]) / 2,
         bbox_list[:, 2] - bbox_list[:, 0],
         bbox_list[:, 3] - bbox_list[:, 1]], axis=-1)
    return cxywh_bbox

# box [xmin, ymin, xmax, ymax]
def xyxy2yxyx(bbox_list):
    return np.stack([bbox_list[:, 1], bbox_list[:, 0],
                     bbox_list[:, 3], bbox_list[:, 2]], axis=-1)

# def bbox_area(box):
#     return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

# def bbox_intersec_area(box_1, box_2, align=False):
#     """ Compute intersection area of two bbox
        
#         Args:
#             box_1, box_2 (list): bounding box pair
#             align (bool): Whether align the center of bbox or not 

#         Returns:
#             intersection area of two bbox (float)
#     """
#     box = [box_2]
#     box.append(box_1)
#     box = np.array(box) 
#     if not align:
#         ibox = np.append(np.amax(np.array(box[:, :2]), axis=0),
#                          np.amin(np.array(box[:, 2:]), axis=0))
#         iw = ibox[2] - ibox[0] + 1
#         ih = ibox[3] - ibox[1] + 1
#         return 0 if iw < 0 or ih < 0 else iw*ih
#     else:
#         width = np.amin(box[:, 2] - box[:, 0]) + 1
#         height = np.amin(box[:, 3] - box[:, 1]) + 1
#         return width * height
        
# # def bbox_intersec(box_1, box_2):
# #     box_2 = [box_2]
# #     box_2.append(box_1)
# #     box_2 = np.array(box_2)
# #     return np.append(np.amax(np.array(box_2[:, :2]), axis=0),
# #                      np.amin(np.array(box_2[:, 2:]), axis=0))

# def bbox_IOU(box_1, box_2, align=False):
#     """ Compute IOU between two bbox
        
#         Args:
#             box_1, box_2 (list): bounding box pair
#             align (bool): Whether align the center of bbox or not 

#         Returns:
#             IOU between two bbox (float)
#     """
#     inter_area = bbox_intersec_area(box_1, box_2, align=align)
#     return inter_area / (bbox_area(box_1) + bbox_area(box_2) - inter_area)

def rescale_bbox(bbox_list, from_shape, to_shape):
    # from_shape, to_shape [h, w]
    # bbox_list[i] [x, y, x, y] or [cx, xy, w, h]
    rescale_bbox = np.stack(
        [bbox_list[:, 0] / from_shape[1] * to_shape[1],
         bbox_list[:, 1] / from_shape[0] * to_shape[0],
         bbox_list[:, 2] / from_shape[1] * to_shape[1],
         bbox_list[:, 3] / from_shape[0] * to_shape[0]], axis=-1)
    return rescale_bbox

def bbox_list_IOU(bbox_list_1, bbox_list_2, align=True):
    # box [xmin, ymin, xmax, ymax]
    bbox_list_1 = np.array(bbox_list_1)
    bbox_list_2 = np.array(bbox_list_2)
    if len(bbox_list_1.shape) == 1:
        bbox_list_1 = [bbox_list_1]
    elif len(bbox_list_1.shape) > 2:
        raise ValueError('Incorrect shape of bbox_list_1')

    if len(bbox_list_2.shape) == 1:
        bbox_list_2 = [bbox_list_2]
    elif len(bbox_list_2.shape) > 2:
        raise ValueError('Incorrect shape of bbox_list_2')

    transpose_sign = False
    if len(bbox_list_2) < len(bbox_list_1):
        bbox_list_1, bbox_list_2 = bbox_list_2, bbox_list_1
        transpose_sign = True

    if align:
        h_list = bbox_list_2[:, 3] - bbox_list_2[:, 1]
        w_list = bbox_list_2[:, 2] - bbox_list_2[:, 0]
        area_list = np.multiply(h_list, w_list)

        iou_list = []
        for bbox in bbox_list_1:
            h = bbox[3] - bbox[1]
            w = bbox[2] - bbox[0]
            area = h * w

            inter_h = np.minimum(h, h_list)
            inter_w = np.minimum(w, w_list)
            inter_area = np.multiply(inter_h, inter_w)
            iou = inter_area / (area_list + area - inter_area)
            iou_list.append(iou)        
    else:
        iou_list = []
        for bbox in bbox_list_1:
            h = bbox[3] - bbox[1]
            w = bbox[2] - bbox[0]
            area = h * w

            h_list = bbox_list_2[:, 3] - bbox_list_2[:, 1]
            w_list = bbox_list_2[:, 2] - bbox_list_2[:, 0]
            area_list = h_list * w_list

            inter_min = np.maximum(bbox[:2], bbox_list_2[:, :2])
            inter_max = np.minimum(bbox[2:], bbox_list_2[:, 2:])
            
            inter_w = inter_max[:, 0] - inter_min[:, 0]
            inter_h = inter_max[:, 1] - inter_min[:, 1]

            inter_area = inter_w * inter_h
            # print(np.where(inter_w < 0) or np.where(inter_h < 0))
            # print(np.any([inter_w < 0, inter_h < 0], axis=0))
            # print(np.where(inter_h < 0))
            inter_area[np.any([inter_w < 0, inter_h < 0], axis=0)] = 0
            # print(area_list,area,inter_area)
            iou = inter_area / (area_list + area - inter_area)
            iou_list.append(iou) 

    iou_list = np.array(iou_list)
    if transpose_sign:
        return iou_list.transpose()
    else:
        return iou_list


if __name__ == '__main__':
    b_1 = [[0, 2, 1, 3], [0, 2, 1, 3], [0, 5, 7, 10]] 
    b_2 = [[0, 2+5.5, 1+5.5, 3+5.5], [0, 2+0.5, 1+0.5, 3+0.5]]
    # b_2 = [i + 0.5 for i in b_2]
    # print(bbox_IOU(b_1, b_2, align=True))

    (bbox_list_IOU(b_1, b_2, align=False))

