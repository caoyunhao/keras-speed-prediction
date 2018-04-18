# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/14 16:47
# @Author  : Yunhao Cao
# @File    : tool.py
import codecs
import os
import cv2
import numpy as np

import config

__author__ = 'Yunhao Cao'

LV_LIST = [
    0.1,
    10,
    25,
    45,
    1 << 32,
]

ORIGIN_SHAPE = (375, 1242, 3)
CUT_SHAPE_0 = (300, 1200, 3)  # 1：4
CUT_SHAPE_1 = (30, 120, 3)  # 1：4
CUT_SHAPE_2 = (75, 300, 3)  # 1：4

TRAIN_SHAPE = CUT_SHAPE_2

compare_path = os.path.join
TRAINSET_DIR = config.TRAINSET_DIR
VALIDATION_DIR = config.VALIDATION_DIR

train_set_dir = TRAINSET_DIR
validation_set_dir = VALIDATION_DIR
sync_name = 'sync.txt'


def TRAIN_SET_DIR_i(i):
    return compare_path(TRAINSET_DIR, 'set_' + i)


def train_set_images(i):
    return compare_path(TRAIN_SET_DIR_i(i), 'images')


def get_all(dir_: str) -> list:
    return list(sorted([os.path.join(dir_, name) for name in os.listdir(dir_) if not name.startswith('.')]))


def read_text(filename):
    with codecs.open(filename, "r", 'utf8') as f:
        return [_.split('\n')[0] for _ in f.readlines() if _]


def write_text(filename, lines, end='\n'):
    with codecs.open(filename, 'w', 'utf8') as f:
        f.write(end.join([str(_) for _ in lines]))


def read_image(filename):
    print('[read_image]', filename)
    return cv2.imread(filename)


def image_cut(data, shape):
    return cv2.resize(data, shape, interpolation=cv2.INTER_CUBIC)


def image_save(filename, data):
    cv2.imwrite(filename, data)


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


if __name__ == '__main__':
    pass
