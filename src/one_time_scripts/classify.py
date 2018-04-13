# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/12 22:23
# @Author  : Yunhao Cao
# @File    : classify.py
import os

from src.common import dir_util, data_util
from src.common import file_util
from src import train_set

__author__ = 'Yunhao Cao'

__all__ = [
    '',
]

origin_dataset_dir = dir_util.origin_dataset_dir

train_set_dir = dir_util.trainset_dir

validation_set_dir = dir_util.validation_dir


def to_name(i):
    i = str(i)
    return '{}{}{}'.format(''.join(['0' for i in range(0, 10 - len(i))]), i, '.png')


def copy():
    for i, set_dirname in enumerate(os.listdir(origin_dataset_dir)):
        if set_dirname.startswith('.'):
            continue
        lines = file_util.read_text(os.path.join(origin_dataset_dir, set_dirname, 'sync.txt'))
        for image_index, line in enumerate(lines):
            target_path = os.path.join(train_set_dir, line)
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            origin_filename = os.path.join(origin_dataset_dir, set_dirname, 'images', to_name(image_index))
            target_filename = os.path.join(target_path, set_dirname + '_' + to_name(image_index))
            print("From {}\n\tTo: {}".format(origin_filename, target_filename))
            # shutil.copyfile(origin_filename, target_filename)
            data = file_util.read_image(origin_filename)
            data = data_util.ArrayCut(data, train_set.CUT_SHAPE_0[:2], mode=8)
            data = file_util.image_cut(data, (train_set.TRAIN_SHAPE[1], train_set.TRAIN_SHAPE[0]))
            file_util.image_save(target_filename, data)
            # break
        # break


def to_validation():
    for i, cate_dirname in enumerate(os.listdir(train_set_dir)):
        if cate_dirname.startswith('.'):
            continue
        cate_dir = os.path.join(train_set_dir, cate_dirname)
        cate_listdir = list(filter(lambda x: not x.startswith('.'), os.listdir(cate_dir)))

        n = len(cate_listdir) // 5

        target_path = os.path.join(validation_set_dir, cate_dirname)

        if not os.path.exists(target_path):
            os.makedirs(target_path)

        for i in range(n):
            os.rename(os.path.join(cate_dir, cate_listdir[i]),
                      os.path.join(target_path, cate_listdir[i]))


if __name__ == '__main__':
    copy()
    to_validation()
