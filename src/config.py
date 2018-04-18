# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/18 10:47
# @Author  : Yunhao Cao
# @File    : config.py

__author__ = 'Yunhao Cao'

# Train config
batch_size = 240
epochs = 20

# path config
ORIGIN_DATA_DIR = ''
TRAINSET_DIR = ''
VALIDATION_DIR = ''

LV_LIST = [
    0.1,
    10,
    25,
    45,
    1 << 32,
]

NUM_OF_LEVEL = len(LV_LIST)

rate = 0.8

ORIGIN_SHAPE = (375, 1242, 3)
CUT_SHAPE_0 = (300, 1200, 3)  # 1：4
CUT_SHAPE_1 = (30, 120, 3)  # 1：4
CUT_SHAPE_2 = (75, 300, 3)  # 1：4

TRAIN_SHAPE = CUT_SHAPE_2
