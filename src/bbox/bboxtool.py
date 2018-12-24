#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bboxtool.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np


# box [xmin, ymin, xmax, ymax]
def bbox_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

def bbox_intersec_area(box_1, box_2, align=False):
    """ Compute intersection area of two bbox
        
        Args:
            box_1, box_2 (list): bounding box pair
            align (bool): Whether align the center of bbox or not 

        Returns:
            intersection area of two bbox (float)
    """
    box = [box_2]
    box.append(box_1)
    box = np.array(box) 
    if not align:
        ibox = np.append(np.amax(np.array(box[:, :2]), axis=0),
                         np.amin(np.array(box[:, 2:]), axis=0))
        iw = ibox[2] - ibox[0] + 1
        ih = ibox[3] - ibox[1] + 1
        return 0 if iw < 0 or ih < 0 else iw*ih
    else:
        width = np.amin(box[:, 2] - box[:, 0]) + 1
        height = np.amin(box[:, 3] - box[:, 1]) + 1
        return width * height
        
# def bbox_intersec(box_1, box_2):
#     box_2 = [box_2]
#     box_2.append(box_1)
#     box_2 = np.array(box_2)
#     return np.append(np.amax(np.array(box_2[:, :2]), axis=0),
#                      np.amin(np.array(box_2[:, 2:]), axis=0))

def bbox_IOU(box_1, box_2, align=False):
    """ Compute IOU between two bbox
        
        Args:
            box_1, box_2 (list): bounding box pair
            align (bool): Whether align the center of bbox or not 

        Returns:
            IOU between two bbox (float)
    """
    inter_area = bbox_intersec_area(box_1, box_2, align=align)
    return inter_area / (bbox_area(box_1) + bbox_area(box_2) - inter_area)


if __name__ == '__main__':
    b_1 = [0, 2, 1, 3]
    b_2 = [0, 2, 1, 3]
    b_2 = [i + 0.5 for i in b_2]
    print(bbox_IOU(b_1, b_2, align=True))
