# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/5 11:31
# @Author  : Yunhao Cao
# @File    : train.py
import codecs
import json
import os
import time

from keras import backend as K, Input, Model, models, Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Flatten, Conv2D, Activation, Lambda, MaxPooling2D, LSTM
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import CustomObjectScope
import tensorflow as tf
import numpy as np

import config

__author__ = 'Yunhao Cao'

# Train config
batch_size = config.batch_size
epochs = config.epochs

# Train old model.(selected model)
use_selected_model = config.USE_SELECTED_MODEL
selected_model_path = config.SELECTED_MODEL_PATH

print("batch_size         :", selected_model_path)
print("epochs             :", selected_model_path)
print("use_selected_model :", use_selected_model)
print("selected_model_path:", selected_model_path)
print('Waiting...(5s)')
time.sleep(5)

# dir config
TRAINSET_DIR = config.TRAINSET_DIR
VALIDATION_DIR = config.VALIDATION_DIR
# shape config
TRAIN_SHAPE = config.TRAIN_SHAPE

num_classes = config.NUM_OF_LEVEL
classes = config.CLASSES
img_width, img_height = TRAIN_SHAPE[:2]

# #########################
# output config
start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(int(time.time())))
saved_path = os.path.join('.', 'saved_model', start_time)
model_name = os.path.join(saved_path, 'model.h5')
history_name = os.path.join(saved_path, 'history.json')

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# #########################
# Model define
def NVIDA():
    inputs = Input(shape=input_shape)
    conv_1 = Conv2D(24, (5, 5), activation="relu", name="conv_1", strides=(2, 2))(inputs)
    conv_2 = Conv2D(36, (5, 5), activation="relu", name="conv_2", strides=(2, 2))(conv_1)
    conv_3 = Conv2D(48, (5, 5), activation='relu', name='conv_3', strides=(2, 2))(conv_2)
    conv_3 = Dropout(.5)(conv_3)

    conv_4 = Conv2D(64, (3, 3), activation="relu", name="conv_4", strides=(1, 1))(conv_3)
    conv_5 = Conv2D(64, (3, 3), activation="relu", name="conv_5", strides=(1, 1))(conv_4)

    flat = Flatten()(conv_5)

    dense_1 = Dense(1164)(flat)
    dense_1 = Dropout(.5)(dense_1)
    dense_2 = Dense(100, activation='relu')(dense_1)
    dense_2 = Dropout(.5)(dense_2)
    dense_3 = Dense(50, activation='relu')(dense_2)
    dense_3 = Dropout(.5)(dense_3)
    dense_4 = Dense(10, activation='relu')(dense_3)
    dense_4 = Dropout(.5)(dense_4)

    final = Dense(num_classes, activation=tf.atan)(dense_4)
    # angle = Lambda(lambda x: tf.mul(tf.atan(x), 2))(final)

    model = Model(inputs, final)
    model.compile(
        optimizer=SGD(lr=.001, momentum=.9),
        loss='mse',
        metrics=['accuracy'],
    )

    return model


def model_v1():
    model = Sequential()

    model.add(Conv2D(8, 3, 3, init='normal', border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(num_classes))

    model.compile(
        optimizer=SGD(lr=.001, momentum=.9),
        loss='mse',
        metrics=['accuracy'],
    )

    # model.summary()

    return model


def model_v2():
    inputs = Input(shape=input_shape)

    conv_1 = Conv2D(8, (3, 3), init='normal', padding='valid',
                    activation="relu", name="conv_1", strides=(2, 2))(inputs)
    pool_1 = MaxPooling2D((2, 2), padding='valid')(conv_1)
    pool_1 = Dropout(0.2)(pool_1)

    flat = Flatten()(pool_1)

    final = Dense(num_classes, activation='softmax')(flat)

    model = Model(inputs, final)
    model.compile(
        optimizer=SGD(lr=.001, momentum=.9),
        loss='mse',
        metrics=['accuracy'],
    )

    return model


def model_v3():
    inputs = Input(shape=input_shape)

    conv_1 = Conv2D(256, (3, 3), kernel_initializer='normal', padding='valid',
                    activation="relu", name="conv_1", strides=(1, 1))(inputs)
    conv_1 = Dropout(0.2)(conv_1)

    conv_2 = Conv2D(128, (3, 3), kernel_initializer='normal', padding='valid',
                    activation="relu", name="conv_2", strides=(1, 1))(conv_1)
    conv_2 = MaxPooling2D((2, 2), padding='valid')(conv_2)
    conv_2 = Dropout(0.2)(conv_2)

    conv_3 = Conv2D(32, (3, 3), kernel_initializer='normal', padding='valid',
                    activation="relu", name="conv_3", strides=(1, 1))(conv_2)
    conv_3 = Dropout(0.2)(conv_3)

    conv_4 = Conv2D(16, (3, 3), kernel_initializer='normal', padding='valid',
                    activation="relu", name="conv_4", strides=(1, 1))(conv_3)
    conv_4 = MaxPooling2D((2, 2), padding='valid')(conv_4)
    conv_4 = Dropout(0.2)(conv_4)

    flat = Flatten()(conv_4)

    final = Dense(num_classes, activation='softmax')(flat)

    model = Model(inputs, final)
    model.compile(
        optimizer=SGD(lr=.001, momentum=.9),
        loss='mse',
        metrics=['accuracy'],
    )

    return model


def get_model():
    if use_selected_model:
        with CustomObjectScope({
            'atan': tf.atan,
        }):
            model = models.load_model(selected_model_path)
        return model
    else:
        return model_v3()


# #########################
# Train set data generator
def data_generator():
    """
    :return: (x, y) Train and validation set data generator
    """
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        TRAINSET_DIR,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
    )
    validation_generator = test_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
    )

    return train_generator, validation_generator


def _main():
    os.makedirs(saved_path)
    model = get_model()

    train_generator, validation_generator = data_generator()

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(saved_path, "weights.{epoch:02d}-{val_acc:.12f}.hdf5"),
        verbose=1,
        save_best_only=True
    )
    lr_plateau = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_lr=0.000001, verbose=1, mode=min)
    # monitor = RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=batch_size,
        callbacks=[checkpoint, lr_plateau],
    )

    with codecs.open(history_name, 'wb', 'utf8') as fp:
        json.dump(dict((k, np.array(v).tolist()) for k, v in history.history.items()), fp)

    model.save(model_name)

    print('Done.')


if __name__ == '__main__':
    _main()
