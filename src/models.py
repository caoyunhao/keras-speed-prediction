# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/6 13:21
# @Author  : Yunhao Cao
# @File    : models.py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

__author__ = 'Yunhao Cao'

__all__ = [
    'CNN',
    'CapsNet',
]


def CNN(input_shape, num_classes):
    """
    user-defined CNN model
    :param input_shape:
    :param data_format: `channels_first` or `channels_last`
    :param num_classes:
    :return:
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape,
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def CapsNet(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(256, (9, 9), padding='same',
                     input_shape=input_shape,
                     activation='relu'))
    model.add(Conv2D(256, (9, 9), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model
