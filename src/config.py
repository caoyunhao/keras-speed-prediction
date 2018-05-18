# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/18 10:47
# @Author  : Yunhao Cao
# @File    : config.py
import os

__author__ = 'Yunhao Cao'

path_join = os.path.join

# Train config
batch_size = 120
epochs = 20

ORIGIN_DATA_DIR = ''

# path config
PROJECT_ROOT = ''
PROJECT_SRC = path_join(PROJECT_ROOT, 'src')
PROJECT_DATASET = path_join(PROJECT_ROOT, 'dataset')

# dir config
PROCESSED_SET_DIR = path_join(PROJECT_DATASET, 'processed_set')
TRAINSET_DIR = path_join(PROJECT_DATASET, 'train_set')
VALIDATION_DIR = path_join(PROJECT_DATASET, 'validation_set')
TEST_DIR = path_join(PROJECT_DATASET, 'test_set')

SAVED_MODEL_DIR = path_join(PROJECT_SRC, 'saved_model')

# Train old model.(selected model)
USE_SELECTED_MODEL = False

SELECTED_MODEL_TIME = '20180510_141941'
SELECTED_WEIGHT = 'weights.20-0.863700788228.hdf5'

SELECTED_MODEL_DIR = path_join(SAVED_MODEL_DIR, SELECTED_MODEL_TIME)
SELECTED_MODEL_JSON = path_join(SELECTED_MODEL_DIR, 'model.json')
SELECTED_MODEL_PATH = path_join(SELECTED_MODEL_DIR, 'model.h5')

# server path config
MODEL_PATH = SELECTED_MODEL_PATH
WEIGHT_PATH = path_join(SELECTED_MODEL_DIR, SELECTED_WEIGHT)
STATIC_PATH = path_join(PROJECT_SRC, 'static')

LV_LIST = [
    5,
    25,
    45,
    1 << 32,
]

CLASSES = ['0', '1', '2', '3']

NUM_OF_LEVEL = len(LV_LIST)

VALIDATION_RATE = 0.1
TEST_RATE = 0.1

ORIGIN_SHAPE = (375, 1242, 3)
CUT_SHAPE_0 = (300, 1200, 3)  # 1：4
CUT_SHAPE_1 = (30, 120, 3)  # 1：4
CUT_SHAPE_2 = (75, 300, 3)  # 1：4

TRAIN_SHAPE = CUT_SHAPE_2

IMAGE_WIDTH = TRAIN_SHAPE[1]
IMAGE_HEIGHT = TRAIN_SHAPE[0]
