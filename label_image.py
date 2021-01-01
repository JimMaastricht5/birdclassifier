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
#
# Revisions by JimMaastricht5@gmail.com
# refactored for object detection and object classification
# blended tensor and tensor flow lite capabilities
# added conversion of cv2 frame to tensor
# added scale rect results from obj detection to apply to full image
# added code for detailed object detection and for general classification
# ==============================================================================
# PY4: pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite  # for pi4 with install wheel above
# import tensorflow as tf  # TF2
# import time


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


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


# initialize tensor flow model
def init_tf2(model_file, num_threads, label_file):
    possible_labels = np.asarray(load_labels(label_file))  # load label file and convert to list
    # interpreter = tf.lite.Interpreter(model_file, num_threads)
    interpreter = tflite.Interpreter(model_file, num_threads)
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
                labelidx = int(det_labels_index[0][index])  # get result label index for labels;
                try:
                    label = labels[labelidx]  # grab text from possible labels
                except:
                    label = ""

                confidences.append(det_confidence)
                confidence_labels.append(label)
                confidence_rects.append(det_rects[0][index])

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

    # if floating_model:  # full tensor bird classification model
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    cindex = np.where(results == np.amax(results))
    lindex = cindex[0]  # grab best result; np array is in max order descending
    try:
        cresult = float(results[cindex])
        lresult = labels[lindex]
    except:
        print('array out of bounds error: confidence and label indices', cindex, lindex)
        print('results', results)
        cresult = float(0)
        lresult = ''

    # cresult = cresult / 100 needed for automl not keras model
    return cresult, lresult  # highest confidence and best label


def convert_cvframe_to_ts(frame, input_details, input_mean, input_std):
    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32
    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    inp_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    reshape_image = inp_img.reshape(width, height, 3)
    image_np_expanded = np.expand_dims(reshape_image, axis=0)

    if floating_model:
        input_data = image_np_expanded.astype('float32')
        input_data = (np.float32(input_data) - input_mean) / input_std
    else:
        input_data = image_np_expanded.astype('uint8')

    return floating_model, input_data


def scale_rect(img, box):
    (img_height, img_width, img_layers) = img.shape
    y_min = int(max(1, (box[0] * img_height)))
    x_min = int(max(1, (box[1] * img_width)))
    y_max = int(min(img_height, (box[2] * img_height)))
    x_max = int(min(img_width, (box[3] * img_width)))
    return (x_min, y_min, x_max, y_max)


def predominant_color(img):
    # from pyimagesearch.com color detection
    # define the list of boundaries.  create sets of Green Blue Red GBR defining lower and upper bounds
    i = 0
    colorcount = {}
    colors = ['Red', 'Blue', 'Yellow', 'Gray']
    boundaries = [
        ([17, 15, 100], [50, 56, 200]),  # Red
        ([86, 31, 4], [220, 88, 50]),  # Blue
        ([25, 146, 190], [62, 174, 250]),  # Yellow
        ([103, 86, 65], [145, 133, 128])  # Gray
    ]

    # loop over the boundaries
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")  # create NumPy arrays from the boundaries
        upper = np.array(upper, dtype="uint8")
        # find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(img, lower, upper)
        maskimg = cv2.bitwise_and(img, img, mask=mask)
        colorcount[i] = maskimg.any(axis=-1).count_nonzero()  # count non-black pixels in image
        i += 1

    cindex = np.where(colorcount == np.amax(colorcount))  # find color with highest count
    color = colors[cindex]
    print(cindex, color)
    return color


# test function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        default='/home/pi/birdclass/test2.jpg',
        help='image to be classified')
    parser.add_argument(
        '-m',
        '--model_file',
        # default='/home/pi/PycharmProjects/pyface2/mobilenet_tweeters.tflite',
        default='/home/pi/birdclass/birdspecies-s-224-93.15.tflite',
        help='.tensor bird classification model to be executed')
    parser.add_argument(
        '-om',
        '--obj_det_file',
        default='/home/pi/birdclass/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite',
        help='.tensor model for obj detection')
    parser.add_argument(
        '-l',
        '--label_file',
        # default='/home/pi/PycharmProjects/pyface2/class_labels.txt',
        default='/home/pi/birdclass/birdspecies-13.txt',
        help='name of file containing labels for bird classification model')
    parser.add_argument(
        '-ol',
        '--obj_det_label_file',
        default='/home/pi/birdclass/lite-model_ssd_mobilenet_v1_1_metadata_2_labelmap.txt',
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
