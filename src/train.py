# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/5 11:31
# @Author  : Yunhao Cao
# @File    : train.py
import os
import sys
import keras
from keras.optimizers import RMSprop

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from src import train_set
from src import models

__author__ = 'Yunhao Cao'

__all__ = [
    '',
]

batch_size = 128
num_classes = train_set.NUM_OF_LEVEL
epochs = 20
# (86, 375, 1242, 3)
img_width = 375
img_height = 1242

model_name = os.path.join('.', 'model.h5')


def _main():
    (x_train, x_test), (y_train, y_test) = train_set.load_data("0001")

    print("x_train.shape :", x_train.shape)
    print("y_train.shape :", y_train.shape)
    print("x_test.shape  :", x_test.shape)
    print("y_test.shape  :", y_test.shape)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = models.CNN(input_shape=(img_width, img_height, 3), num_classes=num_classes, data_format='channels_last')

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    model.save(model_name)

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    _main()
