# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf  # TF2
import json


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def load_json_labels(filename):
    with open(filename) as f:
        labels = json.load(f)
    return labels


def main(args):
    img = Image.open(args.image)
    interpreter, possible_labels = init_tf2(args.model_file, args.num_threads, args.label_file)
    result, label = set_label(img, possible_labels, interpreter, args.input_mean, args.input_std)
    print('result', result)
    print('label', label)


# initialize tensor flow model
def init_tf2(model_file, num_threads, label_file, type="TXT"):
    possible_labels = np.asarray(load_labels(label_file))  # load label file and convert to list
    if type="JSON":
        possible_labels = load_json_labels(label_file)
    else:
        possible_labels = load_labels(label_file)  # load label file
    interpreter = tf.lite.Interpreter(model_file, num_threads)
    interpreter.allocate_tensors()
    return interpreter, possible_labels


# input image and return best result and label
def set_label(img, labels, interpreter, input_mean, input_std):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # *****
    inp_img = cv2.resize(img, (width, height))  # resize to respect input shape and tensor model
    rgb = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)  # convert img to RGB
    # rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)  # TF full tensor
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)  # TF Lite
    input_data = tf.expand_dims(rgb_tensor, 0)  # add dims to RGB tensor
    # *****

    # if floating_model:
    #    input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    for i in top_k:
        if floating_model:  # full tensor
            print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
        else:  # tensor lite
            print('{:08.6f}: {}'.format((float(results[i]) / 255), labels[i]))

    print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
    return results[0], labels[0]  # confidence and best label


def convert_cvframe_to_ts(opencv2, frame):
    numpy_frame = np.asarray(frame)
    numpy_frame = opencv2.normalize(numpy_frame.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
    numpy_final = np.expand_dims(numpy_frame, axis=0)
    return numpy_final


# test function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        default='/home/pi/birdclass/cardinal.jpg',
        help='image to be classified')
    parser.add_argument(
        '-m',
        '--model_file',
        default='/home/pi/birdclass/mobilenet_tweeters.tflite',
        help='.tflite model to be executed')
    parser.add_argument(
        '-l',
        '--label_file',
        default='/home/pi/birdclass/class_labels.txt',
        help='name of file containing labels')
    parser.add_argument(
        '--input_mean',
        default=127.5, type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=127.5, type=float,
        help='input standard deviation')
    parser.add_argument(
        '--num_threads', default=None, type=int, help='number of threads')
    arguments = parser.parse_args()

    main(arguments)
