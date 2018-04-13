# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/5 11:31
# @Author  : Yunhao Cao
# @File    : train.py
import os
import sys
import time

import keras
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from src.common import dir_util
from src import train_set
from src import models

__author__ = 'Yunhao Cao'

__all__ = [
    '',
]

batch_size = 12
num_classes = train_set.NUM_OF_LEVEL
epochs = 20
# (86, 375, 1242, 3)
img_width, img_height = train_set.TRAIN_SHAPE[:2]

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(int(time.time())))


start_time = get_timestamp()
saved_path = os.path.join(
    '.',
    'saved_model',
    start_time,
)

if not os.path.exists(saved_path):
    os.makedirs(saved_path)

model_name = os.path.join(saved_path, 'model.h5')
model_weight = os.path.join(saved_path, 'model_weight.h5')


def get_checkpoint_name():
    return os.path.join(saved_path, get_timestamp() + '.hdf5')


def save(model):
    model.save(model_name)
    model.save_weights(model_weight)


def data_generator():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        dir_util.trainset_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=['0', '1', '2', '3', '4'],
        class_mode='categorical',
    )
    validation_generator = test_datagen.flow_from_directory(
        dir_util.validation_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=['0', '1', '2', '3', '4'],
        class_mode='categorical',
    )

    return train_generator, validation_generator


def train_v1():
    (x_train, x_test), (y_train, y_test) = train_set.load_data("0001")

    print("x_train.shape :", x_train.shape)
    print("y_train.shape :", y_train.shape)
    print("x_test.shape  :", x_test.shape)
    print("y_test.shape  :", y_test.shape)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = models.CNN(
        input_shape=input_shape,
        num_classes=num_classes
    )

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    check_point_file_name = get_checkpoint_name()

    model_checkpoint = ModelCheckpoint(check_point_file_name)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    save(model)

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def train_v2():
    model = models.CapsNet(
        input_shape=input_shape,
        num_classes=num_classes,
    )
    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    train_generator, validation_generator = data_generator()

    # check_point_file_name = get_checkpoint_name()
    # model_checkpoint = ModelCheckpoint(check_point_file_name)

    model.fit_generator(
        train_generator,
        steps_per_epoch=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=validation_generator,
        validation_steps=batch_size
    )
    save(model)


def _main():
    train_v2()


if __name__ == '__main__':
    _main()
