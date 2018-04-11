# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/6 11:07
# @Author  : Yunhao Cao
# @File    : data_util.py
import numpy as np

__author__ = 'Yunhao Cao'

__all__ = [
    'Array',
    'ArraySplit',
    'ArrayCut',
]


def Array(i, dtype=None):
    return np.array(i, dtype=dtype)


def ArraySplit(array, rate):
    n = int(array.shape[0] * rate)
    return array[:n], array[n:]


def ArrayCut(array, out_shape, mode=5):
    """
    1 2 3
    4 5 6
    7 8 9
    :param array: np.ndarray
    :param out_shape: (height, width)
    :param mode: 1~9
    :return: cut array
    """
    in_shape = array.shape[:2]
    y_diff = in_shape[0] - out_shape[0]
    x_diff = in_shape[1] - out_shape[1]
    y_start = x_start = 0
    y_end = out_shape[0]
    x_end = out_shape[1]
    if mode in (4, 5, 6):
        y_start = y_diff // 2
        y_end = in_shape[0] - y_start - y_diff % 2
    if mode in (2, 5, 8):
        x_start = x_diff // 2
        x_end = in_shape[1] - x_start - x_diff % 2
    if mode in (7, 8, 9):
        y_start = y_diff
        y_end = None
    if mode in (3, 6, 9):
        x_start = x_diff
        x_end = None

    return array[y_start:y_end, x_start: x_end]


def _test():
    a = np.array([[i for i in range(s, s + 10)] for s in range(0, 10)], dtype='int32')

    print(type(a))

    print(ArrayCut(a, (3, 3), 9))


if __name__ == '__main__':
    _test()
