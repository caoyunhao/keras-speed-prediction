# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/6 10:55
# @Author  : Yunhao Cao
# @File    : train_set.py
import os
import re
import shutil

import cv2
import numpy as np
import tool
import config

__author__ = 'Yunhao Cao'

__all__ = [
    '',
]

# config
level_list = config.LV_LIST
classes = config.NUM_OF_LEVEL
validation_rate = config.VALIDATION_RATE
test_rate = config.TEST_RATE

origin_data_dir = config.ORIGIN_DATA_DIR
processed_set_dir = config.PROCESSED_SET_DIR
trainset_dir = config.TRAINSET_DIR
validation_set_dir = config.VALIDATION_DIR
test_set_dir = config.TEST_DIR

cut_shape = config.CUT_SHAPE_0
train_shape = config.TRAIN_SHAPE

image_width = config.IMAGE_WIDTH
image_height = config.IMAGE_HEIGHT

# External function
compare_path = tool.compare_path
get_all = tool.get_all
curname = tool.curname


def get_lv(v) -> int:
    """
    返回速度等级
    """
    for i, lv in enumerate(level_list):
        if abs(v) < lv:
            return i


def generate_sync_txt():
    vf = 8  # forward velocity, i.e. parallel to earth-surface (m/s)
    vl = 9  # leftward velocity, i.e. parallel to earth-surface (m/s)
    af = 14  # forward acceleration (m/s^2)

    for dir_ in tool.get_all(origin_data_dir):
        sync_data_dir = compare_path(dir_, 'oxts', 'data')
        print(sync_data_dir)
        txt_list = tool.get_all(sync_data_dir)

        outlines = list()
        for txt in txt_list:
            lines = tool.read_text(txt)
            line_items = lines[0].split()
            # print(float(line_items[vf]) * 3.6)
            v_origin = float(line_items[vf]) * 3.6
            v_level = get_lv(v_origin)
            if v_level is None:
                raise Exception
            item = '{} {}'.format(v_origin, v_level)
            outlines.append(item)

        tool.write_text(compare_path(dir_, tool.sync_name), outlines)


def to_name(i):
    i = str(i)
    return '{}{}{}'.format(''.join(['0' for i in range(0, 10 - len(i))]), i, '.png')


def copy_to_process_set():
    for i, set_dir in enumerate(tool.get_all(origin_data_dir)):

        lines = tool.read_text(compare_path(set_dir, 'sync.txt'))

        set_id = re.match('.*2011_09_26_drive_(?P<set_id>\d*)_sync.*', set_dir).groupdict()["set_id"]

        for image_index, line in enumerate(lines):
            v, level = line.split()
            target_path = compare_path(processed_set_dir, level)
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            origin_filename = compare_path(set_dir, 'image_02', 'data', to_name(image_index))
            target_filename = compare_path(target_path, "set_{}_lv{}_{}".format(set_id, level, to_name(image_index)))
            print("From {}\n\tTo: {}".format(origin_filename, target_filename))
            data = tool.read_image(origin_filename)
            if data is None:
                print('[WAIN] From image_03', set_dir, image_index)
                origin_filename = compare_path(set_dir, 'image_03', 'data', to_name(image_index))
                data = tool.read_image(origin_filename)
            if data is None:
                print("[ERROR] No exists in ", set_dir, image_index)
            else:
                data = tool.ArrayCut(data, cut_shape[:2], mode=8)
                data = tool.image_cut(data, (image_width, image_height))
                tool.image_save(target_filename, data)


def split_by_copy():
    import random
    from_dir = processed_set_dir
    for i, cate_dirname in enumerate(os.listdir(from_dir)):
        if cate_dirname.startswith('.'):
            continue

        cate_dir = compare_path(from_dir, cate_dirname)
        cate_listdir = list(filter(lambda x: not x.startswith('.'), os.listdir(cate_dir)))

        v_t_n = int(len(cate_listdir) * (validation_rate + test_rate))

        v_n = int(v_t_n * validation_rate / (validation_rate + test_rate))

        validation_test_files = random.sample(cate_listdir, v_t_n)

        validation_files = validation_test_files[:v_n]
        test_files = validation_test_files[v_n:]

        validation_cate_path = compare_path(validation_set_dir, cate_dirname)
        print(validation_cate_path)
        if not os.path.exists(validation_cate_path):
            os.makedirs(validation_cate_path)

        for validation_file in validation_files:
            shutil.copy(compare_path(cate_dir, validation_file),
                        compare_path(validation_cate_path, validation_file))

        test_cate_path = compare_path(test_set_dir, cate_dirname)
        print(test_cate_path)
        if not os.path.exists(test_cate_path):
            os.makedirs(test_cate_path)

        for file_name in test_files:
            shutil.copy(compare_path(cate_dir, file_name),
                        compare_path(test_cate_path, file_name))

        train_set_path = compare_path(trainset_dir, cate_dirname)
        if not os.path.exists(train_set_path):
            os.makedirs(train_set_path)

        train_set_files = list(set(cate_listdir).difference(set(validation_files)).difference(set(test_files)))
        for train_set_file in train_set_files:
            shutil.copy(compare_path(cate_dir, train_set_file),
                        compare_path(train_set_path, train_set_file))


def image_to_feature_vector(image, size=(image_width, image_height)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(32, 32, 32)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


def load_data():
    raw_images = []
    features = []
    labels = []

    # loop over the input images
    for c_i, category_dir in enumerate(get_all(processed_set_dir)):
        # load the image and extract the class label
        # our images were named as labels.image_number.format
        label = curname(category_dir)
        for i_i, image_path in enumerate(get_all(category_dir)):
            image = cv2.imread(image_path)

            # extract raw pixel intensity "features"
            # followed by a color histogram to characterize the color distribution of the pixels
            # in the image
            pixels = image_to_feature_vector(image)
            hist = extract_color_histogram(image)

            # add the messages we got to the raw images, features, and labels matricies
            raw_images.append(pixels)
            features.append(hist)
            labels.append(label)

            n = len(get_all(category_dir))
            if i_i > 0 and ((i_i + 1) % 200 == 0 or i_i == n - 1):
                print("[INFO] processed {}/{}".format(i_i + 1, n))
        print("[INFO] `{}` done.({}/{})".format(label, c_i + 1, len(get_all(processed_set_dir))))

    raw_images = np.array(raw_images)
    features = np.array(features)
    labels = np.array(labels)

    print("[INFO] pixels matrix: {:.2f}MB".format(
        raw_images.nbytes / (1024 * 1000.0)))
    print("[INFO] features matrix: {:.2f}MB".format(
        features.nbytes / (1024 * 1000.0)))

    return raw_images, features, labels


def _test():
    pass
    # generate_sync_txt()
    # copy_to_process_set()
    # split_by_copy()


if __name__ == '__main__':
    _test()
