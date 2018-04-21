# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/6 10:55
# @Author  : Yunhao Cao
# @File    : train_set.py
import os
import re

import tool
import config

__author__ = 'Yunhao Cao'

__all__ = [
    '',
]

level_list = config.LV_LIST
classes = config.NUM_OF_LEVEL
rate = config.rate
origin_data_dir = config.ORIGIN_DATA_DIR
trainset_dir = config.TRAINSET_DIR
cut_shape = config.CUT_SHAPE_0
train_shape = config.TRAIN_SHAPE

image_width = config.IMAGE_WIDTH
image_height = config.IMAGE_HEIGHT

compare_path = tool.compare_path


def get_lv(v) -> int:
    """
    返回速度等级
    """
    for i, lv in enumerate(level_list):
        if abs(v) < lv:
            return i


def generate_sync_txt():
    vf = 8
    vl = 9
    af = 14

    for dir_ in tool.get_all(origin_data_dir):
        sync_data_dir = compare_path(dir_, 'oxts', 'data')
        print(sync_data_dir)
        txt_list = tool.get_all(sync_data_dir)

        outlines = list()
        for txt in txt_list:
            lines = tool.read_text(txt)
            line_items = lines[0].split()
            # print(float(line_items[vf]) * 3.6)
            v_origin = float(line_items[vf]) * 3.6
            v_level = get_lv(v_origin)
            if v_level is None:
                raise Exception
            item = '{} {}'.format(v_origin, v_level)
            outlines.append(item)

        tool.write_text(compare_path(dir_, tool.sync_name), outlines)


def to_name(i):
    i = str(i)
    return '{}{}{}'.format(''.join(['0' for i in range(0, 10 - len(i))]), i, '.png')


def copy():
    for i, set_dir in enumerate(tool.get_all(origin_data_dir)):

        lines = tool.read_text(compare_path(set_dir, 'sync.txt'))

        set_id = re.match('.*2011_09_26_drive_(?P<set_id>\d*)_sync.*', set_dir).groupdict()["set_id"]

        for image_index, line in enumerate(lines):
            v, level = line.split()
            target_path = compare_path(tool.train_set_dir, level)
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            origin_filename = compare_path(set_dir, 'image_02', 'data', to_name(image_index))
            target_filename = compare_path(target_path, "set_{}_lv{}_{}".format(set_id, level, to_name(image_index)))
            print("From {}\n\tTo: {}".format(origin_filename, target_filename))
            data = tool.read_image(origin_filename)
            if data is None:
                print('[WAIN] From image_03', set_dir, image_index)
                origin_filename = compare_path(set_dir, 'image_03', 'data', to_name(image_index))
                data = tool.read_image(origin_filename)
            if data is None:
                print("[ERROR] No exists in ", set_dir, image_index)
            else:
                data = tool.ArrayCut(data, cut_shape[:2], mode=8)
                data = tool.image_cut(data, (image_width, image_height))
                tool.image_save(target_filename, data)


def to_validation():
    for i, cate_dirname in enumerate(os.listdir(tool.train_set_dir)):
        if cate_dirname.startswith('.'):
            continue

        cate_dir = compare_path(tool.train_set_dir, cate_dirname)
        cate_listdir = list(filter(lambda x: not x.startswith('.'), os.listdir(cate_dir)))

        n = len(cate_listdir) // 5

        target_path = compare_path(tool.validation_set_dir, cate_dirname)

        if not os.path.exists(target_path):
            os.makedirs(target_path)

        for i in range(n):
            os.rename(compare_path(cate_dir, cate_listdir[i]),
                      compare_path(target_path, cate_listdir[i]))


def _test():
    # print(get_set('0001').shape)
    # print(get_flag('0001').shape)
    # print(tool.dir_util.origin_sync_dirname)
    # generate_sync_txt()
    copy()
    to_validation()


if __name__ == '__main__':
    _test()
