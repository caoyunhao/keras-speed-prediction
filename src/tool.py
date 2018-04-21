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

level_list = config.LV_LIST
input_shape = config.TRAIN_SHAPE

compare_path = os.path.join

train_set_dir = config.TRAINSET_DIR
validation_set_dir = config.VALIDATION_DIR
sync_name = 'sync.txt'


def get_max_index(l):
    max_index = 0
    max_item = 0
    for i, item in enumerate(l):
        if item > max_item:
            max_item = item
            max_index = i

    return max_index


def get_v_range(level):
    start = level_list[level - 1] if level > 0 else 0
    if (level + 1) < len(level_list):
        end = level_list[level]
    else:
        end = None
    return start, end


def TRAIN_SET_DIR_i(i):
    return compare_path(train_set_dir, 'set_' + i)


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
