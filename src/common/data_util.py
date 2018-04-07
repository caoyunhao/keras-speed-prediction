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
]


def Array(i, dtype=None):
    return np.array(i, dtype=dtype)


def ArraySplit(array, rate):
    n = int(array.shape[0] * rate)
    return array[:n], array[n:]


def _test():
    pass


if __name__ == '__main__':
    _test()
