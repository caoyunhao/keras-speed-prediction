# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/6 10:55
# @Author  : Yunhao Cao
# @File    : train_set.py
from src.common import dir_util, file_util, data_util

__author__ = 'Yunhao Cao'

__all__ = [
    'LV_LIST',
    'NUM_OF_LEVEL',
    'get_set',
    'get_flag',
    'load_data',
]

MOVE_CRITICAL = 0.1

LV_LIST = [
    MOVE_CRITICAL,
    10,
    25,
    45,
    1 << 32,
]

NUM_OF_LEVEL = len(LV_LIST)

SET_NUMS = [
    '0001',
    '0002',
    '0005',
    '0009',
    '0011',
]

rate = 0.8


def get_set(i):
    images_dir = dir_util.train_set_images(i)
    tmp_list = list()
    for filename in file_util.get_all(images_dir):
        data = file_util.read_image(filename)
        tmp_list.append(data)

    return data_util.Array(tmp_list)


def get_flag(i):
    sync_file = dir_util.sync_fullname(i)
    lines = file_util.read_text(sync_file)

    return data_util.Array(lines)


def load_data(i):
    return data_util.ArraySplit(get_set(i), rate), data_util.ArraySplit(get_flag(i), rate)


def _test():
    print(get_set('0001').shape)
    print(get_flag('0001').shape)


if __name__ == '__main__':
    _test()
