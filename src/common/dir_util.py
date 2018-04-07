# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/5 15:16
# @Author  : Yunhao Cao
# @File    : dir_util.py
import os
from src import config

__author__ = 'Yunhao Cao'

__all__ = [
    'TRAIN_SET_DIR_i',
    'train_set_images',
    'sync_fullname',
    'origin_syncdata',
]

compare_path = os.path.join

root_dir = config.PROJECT_ROOT
dataset_dirname = 'dataset'
trainset_dirname = 'train_set'
set_prefix = 'set_'
images_dirname = 'images'
origin_sync_dirname = 'sync_data'
sync_txt = 'sync.txt'

dataset_dir = compare_path(root_dir, dataset_dirname)
trainset_dir = compare_path(dataset_dir, trainset_dirname)


def TRAIN_SET_DIR_i(i: str):
    return compare_path(trainset_dir, set_prefix + i)


def train_set_images(i):
    return compare_path(TRAIN_SET_DIR_i(i), images_dirname)


def sync_fullname(i):
    return compare_path(TRAIN_SET_DIR_i(i), sync_txt)


def origin_syncdata(i):
    return compare_path(TRAIN_SET_DIR_i(i), origin_sync_dirname)


def _test():
    pass


if __name__ == '__main__':
    _test()
