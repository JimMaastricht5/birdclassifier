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
# import time
import cv2
import numpy as np
# from PIL import Image
import tensorflow as tf  # TF2


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def main(args):
    img = cv2.imread(args.image)
    obj_interpreter, obj_labels = init_tf2(args.obj_det_file, args.num_threads, args.obj_det_label_file)
    results, labels, rects = object_detection(args.confidence, img, obj_labels,
                                              obj_interpreter, args.input_mean, args.input_std)

    print('objects detected', results)
    print('labels detected', labels)
    print('rectangles', rects)
    interpreter, possible_labels = init_tf2(args.model_file, args.num_threads, args.label_file)
    result, label = set_label(img, possible_labels, interpreter, args.input_mean, args.input_std)
    print('final result', result)
    print('final label', label)


# initialize tensor flow model
def init_tf2(model_file, num_threads, label_file):
    possible_labels = np.asarray(load_labels(label_file))  # load label file and convert to list
    interpreter = tf.lite.Interpreter(model_file, num_threads)
    interpreter.allocate_tensors()
    return interpreter, possible_labels


# input image and return best result and label
def object_detection(min_confidence, img, labels, interpreter, input_mean, input_std):
    confidences = []
    confidence_labels = []
    confidence_rects = []

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    floating_model, input_data = convert_cvframe_to_ts(img, input_details, input_mean, input_std)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # start_time = time.time()
    interpreter.invoke()
    # stop_time = time.time()

    if floating_model is False:  # tensor lite obj det prebuilt model
        det_rects = interpreter.get_tensor(output_details[0]['index'])
        det_labels_index = interpreter.get_tensor(output_details[1]['index'])  # labels are an array for each result
        det_confidences = interpreter.get_tensor(output_details[2]['index'])

        for index, det_confidence in enumerate(det_confidences[0]):
            if det_confidence >= min_confidence:
                labelidx = int(det_labels_index[index][0] - 1)  # get result label index for labels; offset -1 row 0
                label = labels[labelidx]  # grab text from possible labels
                confidences.append(det_confidence)
                confidence_labels.append(label)
                confidence_rects.append(det_rects[index][0])
                # print('confidence and label {:08.6f}: {}'.format(float(det_confidence), label))
                # print("Rectangles are: {}".format(det_rects[index]))

    # print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
    return confidences, confidence_labels, confidence_rects  # confidence and best label


# input image and return best result and label
def set_label(img, labels, interpreter, input_mean, input_std):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    floating_model, input_data = convert_cvframe_to_ts(img, input_details, input_mean, input_std)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # start_time = time.time()
    interpreter.invoke()
    # stop_time = time.time()

    if floating_model:  # full tensor bird classification model
        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)
        # top_k = results.argsort()[-5:][::-1]
        # for i in top_k:
        #     print('{:08.6f}: {}'.format(float(results[i]), labels[i]))

    # print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
    return results[0], labels[0]  # confidence and best label


def convert_cvframe_to_ts(frame, input_details, input_mean, input_std):
    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    inp_img = cv2.resize(frame, (width, height))  # resize to respect input shape and tensor model
    rgb = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)  # convert img to RGB
    if floating_model:
        rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)  # TF full tensor
    else:
        rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)  # TF Lite

    input_data = tf.expand_dims(rgb_tensor, 0)  # add dims to RGB tensor

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    return floating_model, input_data


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
        help='.tensor bird classification model to be executed')
    parser.add_argument(
        '-om',
        '--obj_det_file',
        default='/home/pi/birdclass/ssd_mobilenet_v1_1_metadata_1.tflite',
        help='.tensor model for obj detection')
    parser.add_argument(
        '-l',
        '--label_file',
        default='/home/pi/birdclass/class_labels.txt',
        help='name of file containing labels for bird classification model')
    parser.add_argument(
        '-ol',
        '--obj_det_label_file',
        default='/home/pi/birdclass/mscoco_label_map.txt',
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
    parser.add_argument('-c', '--confidence', type=float, default=0.5)
    arguments = parser.parse_args()

    main(arguments)
