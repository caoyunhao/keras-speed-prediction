# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/5 15:30
# @Author  : Yunhao Cao
# @File    : file_util.py
import codecs
import os

import cv2

__author__ = 'Yunhao Cao'

__all__ = [
    'get_all',
    'read_text',
    'read_image',
    'write_text',
]


def get_all(dir_: str) -> list:
    return [os.path.join(dir_, name) for name in os.listdir(dir_)]


def read_text(filename):
    with codecs.open(filename, "r", 'utf8') as f:
        return [_.split('\n')[0] for _ in f.readlines() if _]


def write_text(filename, lines, end='\n'):
    with codecs.open(filename, 'w', 'utf8') as f:
        f.write(end.join([str(_) for _ in lines]))


def read_image(filename):
    return cv2.imread(filename)


def image_cut(data, shape):
    return cv2.resize(data, shape, interpolation=cv2.INTER_CUBIC)


def image_save(filename, data):
    cv2.imwrite(filename, data)


def _test():
    # print(read_text('./test.txt'))
    print(read_image('./test.png').shape)


if __name__ == '__main__':
    _test()
