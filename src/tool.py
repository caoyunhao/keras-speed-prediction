# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/14 16:47
# @Author  : Yunhao Cao
# @File    : tool.py
import codecs
import json
import os

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import CustomObjectScope

import config

__author__ = 'Yunhao Cao'

batch_size = config.batch_size
epochs = config.epochs

level_list = config.LV_LIST
train_shape = config.TRAIN_SHAPE

train_set_dir = config.TRAINSET_DIR
validation_set_dir = config.VALIDATION_DIR

# test
test_set_dir = config.TEST_DIR
img_width, img_height = train_shape[:2]
classes = config.CLASSES

sync_name = 'sync.txt'

compare_path = os.path.join


def get_max_index(l):
    max_index = 0
    max_item = 0
    for i, item in enumerate(l):
        if item > max_item:
            max_item = item
            max_index = i

    return max_index


def get_v_range(level):
    start = level_list[level - 1] if level > 0 else 0
    if (level + 1) < len(level_list):
        end = level_list[level]
    else:
        end = None
    return start, end


def TRAIN_SET_DIR_i(i):
    return compare_path(train_set_dir, 'set_' + i)


def train_set_images(i):
    return compare_path(TRAIN_SET_DIR_i(i), 'images')


def get_all(dir_: str) -> list:
    return list(sorted([os.path.join(dir_, name) for name in os.listdir(dir_) if not name.startswith('.')]))


def curname(fullpath):
    return fullpath.split(os.path.sep)[-1]


def read_text(filename):
    with codecs.open(filename, "r", 'utf8') as f:
        return [_.split('\n')[0] for _ in f.readlines() if _]


def write_text(filename, lines, end='\n'):
    with codecs.open(filename, 'w', 'utf8') as f:
        f.write(end.join([str(_) for _ in lines]))


def read_image(filename):
    print('[read_image]', filename)
    return cv2.imread(filename)


def image_cut(data, shape):
    return cv2.resize(data, shape, interpolation=cv2.INTER_CUBIC)


def image_save(filename, data):
    cv2.imwrite(filename, data)


def Array(i, dtype=None):
    return np.array(i, dtype=dtype)


def ArraySplit(array, rate):
    n = int(array.shape[0] * rate)
    return array[:n], array[n:]


def ArrayCut(array, out_shape, mode=5):
    """
    1 2 3
    4 5 6
    7 8 9
    :param array: np.ndarray
    :param out_shape: (height, width)
    :param mode: 1~9
    :return: cut array
    """
    in_shape = array.shape[:2]
    y_diff = in_shape[0] - out_shape[0]
    x_diff = in_shape[1] - out_shape[1]
    y_start = x_start = 0
    y_end = out_shape[0]
    x_end = out_shape[1]
    if mode in (4, 5, 6):
        y_start = y_diff // 2
        y_end = in_shape[0] - y_start - y_diff % 2
    if mode in (2, 5, 8):
        x_start = x_diff // 2
        x_end = in_shape[1] - x_start - x_diff % 2
    if mode in (7, 8, 9):
        y_start = y_diff
        y_end = None
    if mode in (3, 6, 9):
        x_start = x_diff
        x_end = None

    return array[y_start:y_end, x_start: x_end]


def save_history(history, history_name):
    with codecs.open(history_name, 'wb', 'utf8') as fp:
        json.dump(dict((k, np.array(v).tolist()) for k, v in history.history.items()), fp)


def save_model(model, model_name, model_json=None):
    model.save(model_name)
    if model_json:
        with codecs.open(model_json, 'wb', 'utf8') as fp:
            fp.write(model.to_json())


def image_generator(path):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True
    )
    generator = datagen.flow_from_directory(
        path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
    )
    return generator


def save_model_json(model, json_path):
    with codecs.open(json_path, 'wb', 'utf8') as fp:
        fp.write(model.to_json())


def load_model(model_path):
    with CustomObjectScope({
        'atan': tf.atan,
    }):
        model = keras.models.load_model(model_path)

    model._make_predict_function()

    return model


def test_model(model):
    generator = image_generator(test_set_dir)
    score = model.evaluate_generator(generator, steps=20)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def history2csv(model_dir):
    import json

    history_json = compare_path(model_dir, 'history.json')
    history_csv = compare_path(model_dir, 'history.csv')

    with codecs.open(history_json, 'r', 'utf8') as f:
        obj = json.load(f)

    def item(k):
        return ','.join([k, ] + list(map(str, obj[k])))

    epochs_line = ','.join(['epochs'] + list(map(str, range(1, epochs + 1))))

    line_all = '\n'.join([
        epochs_line,
        item('val_acc'),
        item('val_loss'),
        item('acc'),
        item('loss'),
    ])

    with codecs.open(history_csv, 'wb', 'utf8') as f:
        f.writelines(line_all)


if __name__ == '__main__':
    # test_model(load_model(config.SELECTED_MODEL_PATH))
    # save_model_json(load_model(config.SELECTED_MODEL_PATH), config.SELECTED_MODEL_JSON)
    history2csv(config.SELECTED_MODEL_DIR)
