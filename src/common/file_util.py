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
    if not dir_.endswith("/"):
        dir_ = dir_ + '/'
    return ["{}{}".format(dir_, name) for name in os.listdir(dir_)]


def read_text(filename):
    with codecs.open(filename, "r", 'utf8') as f:
        return [_.split('\n')[0] for _ in f.readlines() if _]


def write_text(filename, lines, end='\n'):
    with codecs.open(filename, 'w', 'utf8') as f:
        f.write(end.join([str(_) for _ in lines]))


def read_image(filename):
    data = cv2.imread(filename)
    data.astype("float32")
    return data / 255


def _test():
    print(read_text('./test.txt'))


if __name__ == '__main__':
    _test()
