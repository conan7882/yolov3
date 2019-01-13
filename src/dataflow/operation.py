#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: operation.py
# Author: Qian Ge <geqian1001@gmail.com>

import copy
import numpy as np
from src.dataflow.base import DataFlow


def suffle_list_array(input_list):
    """ Suffle list of np.array in-place. Each array shuffled by the same order.

        Args:
            input_list (list of np.array): each array must has the same length

        Return:
            Suffled list of np.array
    """
    idxs = np.arange(len(input_list[0]))
    np.random.shuffle(idxs)
    for idx, array in enumerate(input_list):
        input_list[idx] = array[idxs]

    return input_list

def slice_list_array(input_list, start, end):
    """ slice list of np.array """

    return_list = []
    for idx, array in enumerate(input_list):
        return_list.append(array[start:end])
    return return_list

def divide_dataflow(dataflow, divide_list, shuffle=True):
    """ divide a DataFlow object into several DataFlow objects

        Args:
            dataflow (DataFlow): Input dataflow to be divided
            divide_list (list): List of percentage for division.
                Each elements within the range (0, 1) and sum of the list must be less than 1. 
            shuffle (bool): whether shuffle or not before division

        Returns:
            List of DataFlow objects with number of samples corresponding
            to the percentage indicated in divide_list
    """

    assert isinstance(dataflow, DataFlow)
    if isinstance(divide_list, (float, int)):
        divide_list = list(divide_list)
    assert sum(divide_list) <= 1
    assert np.amin(divide_list) > 0
    assert np.amax(divide_list) < 1.

    n_flow = len(divide_list)
    file_name_list = dataflow.get_file_name_list()
    n_total_sample = dataflow.size()
    if shuffle:
        file_name_list = suffle_list_array(file_name_list)
        
    dataflow_list = []
    start_id = 0
    for flow_id in range(n_flow - 1):
        df = copy.deepcopy(dataflow)
        n_sample = int(divide_list[flow_id] * n_total_sample)
        
        df_file_name = slice_list_array(file_name_list, start_id, start_id + n_sample)
        df.reset_file_name_list(df_file_name)
        dataflow_list.append(df)
        start_id += n_sample

    n_sample = int(divide_list[-1] * n_total_sample)
    df_file_name = slice_list_array(file_name_list, start_id, start_id + n_sample)
    dataflow.reset_file_name_list(df_file_name)
    dataflow_list.append(dataflow)

    print('Divided into {} dataflows:'.format(len(divide_list)), end='')
    for idx, df in enumerate(dataflow_list):
        print('[{}] {} samples '.format(idx, df.size()), end='')
    print('.')

    return dataflow_list



