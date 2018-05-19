# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/18 18:01
# @Author  : Yunhao Cao
import os
import time

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import train_set

__author__ = 'Yunhao Cao'

start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(int(time.time())))
saved_path = os.path.join('.', 'saved_model', start_time)
model_name = os.path.join(saved_path, '1.model')

# External function
load_data = train_set.load_data

raw_images, features, labels = load_data()

(trainRI, testRI, trainRL, testRL) = train_test_split(
    raw_images, labels, test_size=0.1, random_state=42)

(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
    features, labels, test_size=0.1, random_state=42)


def SVC_model():
    model = SVC(max_iter=1000, class_weight='balanced')

    print("[INFO] Using SVC model...")
    return model


if __name__ == '__main__':
    print("[INFO] evaluating raw pixel accuracy...")
    model = SVC_model()

    print('[INFO] Train start.')
    model.fit(trainRI, trainRL)
    print('[INFO] Train done.')

    joblib.dump(model, model_name)
    print('[INFO] Save done.')

    acc = model.score(testRI, testRL)
    print("[INFO] SVM-SVC raw pixel accuracy: {:.2f}%".format(acc * 100))
