# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/20 10:17
# @Author  : Yunhao Cao
# @File    : server.py
import os
import time
import uuid

from flask import Flask, request, render_template, url_for

import cv2
import tool
import config

__author__ = 'Yunhao Cao'

__all__ = [
    '',
]

model_path = config.MODEL_PATH
weight_path = config.WEIGHT_PATH
input_shape = config.TRAIN_SHAPE
image_width = config.IMAGE_WIDTH
image_height = config.IMAGE_HEIGHT
static_path = config.STATIC_PATH

resize = tool.image_cut
load_model = tool.load_model

static_images_path = os.path.join(static_path, 'images')


def get_time():
    t = int(time.time())
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + '_' + str(t)


def get_receive_name():
    return "{}_{}.png".format(get_time(), uuid.uuid4())


def v_message(v_range, v_level):
    start = v_range[0]
    end = v_range[1]
    if start == 0:
        return '建议速度 0~{} km/h. ({})'.format(end, v_level)
    elif end is None:
        return '建议速度大于 {} km/h. ({})'.format(start, v_level)
    else:
        return '建议速度 {}~{} km/h. ({})'.format(start, end, v_level)


app = Flask(__name__)

app.config.update(dict(
    DEBUG=True,
    SECRET_KEY=b'_5#y2L"F4Q8z\n\xec]/',
))


@app.route('/', methods=['GET', ])
def hello():
    return "Hello world !"


@app.route('/predict', methods=['GET', 'POST', ])
def predict():
    predict_result_msg = None
    image_name = ''
    if request.method == 'POST':
        try:
            image = request.files['file']
        except:
            image = None

        try:
            cv = request.form['cv']
        except:
            cv = None

        if cv is not None:
            msg = '当前速度: {}km/s. '.format(cv)
        else:
            msg = ''

        if image:
            model = load_model(model_path)

            image_name = get_receive_name()
            image_static_path = os.path.join(static_images_path, image_name)

            image.save(image_static_path)
            time.sleep(0.5)

            data = cv2.imread(image_static_path) / 255
            data = resize(data, (image_width, image_height))

            ret = model.predict(data[None, :, :, :], batch_size=1, verbose=1)

            print(ret)

            v_level = tool.get_max_index(list(ret[0]))
            v_range = tool.get_v_range(v_level)

            msg += v_message(v_range, v_level)

            predict_result_msg = msg

    return render_template('predict.html', predict_result_msg=predict_result_msg, image_url=image_url(image_name))


def image_url(name):
    return url_for('static', filename="images/" + name)


def _main():
    if not os.path.exists(static_images_path):
        os.makedirs(static_images_path)
    app.run(host="0.0.0.0")


if __name__ == '__main__':
    _main()
