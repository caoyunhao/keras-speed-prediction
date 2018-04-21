# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/18 10:47
# @Author  : Yunhao Cao
# @File    : config.py
import os

__author__ = 'Yunhao Cao'

path_join = os.path.join

# Train config
batch_size = 12
epochs = 10

ORIGIN_DATA_DIR = ''

# path config
PROJECT_ROOT = ''
PROJECT_SRC = path_join(PROJECT_ROOT, 'src')
PROJECT_DATASET = path_join(PROJECT_ROOT, 'dataset')

TRAINSET_DIR = path_join(PROJECT_DATASET, 'train_set')
VALIDATION_DIR = path_join(PROJECT_DATASET, 'validation_set')

# server
MODEL_PATH = path_join(PROJECT_SRC, 'saved_model', '20180415_154216', 'model.h5')
WEIGHT_PATH = path_join(PROJECT_SRC, 'saved_model', '20180415_154216', 'weights.20-0.324324325073.hdf5')
STATIC_PATH = path_join(PROJECT_SRC, 'static')

LV_LIST = [
    5,
    25,
    45,
    1 << 32,
]

CLASSES = ['0', '1', '2', '3']

NUM_OF_LEVEL = len(LV_LIST)

rate = 0.8

ORIGIN_SHAPE = (375, 1242, 3)
CUT_SHAPE_0 = (300, 1200, 3)  # 1：4
CUT_SHAPE_1 = (30, 120, 3)  # 1：4
CUT_SHAPE_2 = (75, 300, 3)  # 1：4

TRAIN_SHAPE = CUT_SHAPE_2

IMAGE_WIDTH = TRAIN_SHAPE[1]
IMAGE_HEIGHT = TRAIN_SHAPE[0]
