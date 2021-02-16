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
try:
    import tflite_runtime.interpreter as tflite  # for pi4 with install wheel above
except:
    import tensorflow as tf  # TF2 for desktop testing


def main(args):
    img = cv2.imread(args.image)
    obj_interpreter, obj_labels = init_tf2(args.obj_det_model, args.numthreads, args.obj_det_labels)
    results, labels, rects = object_detection(args.bconfidence, img, obj_labels,
                                              obj_interpreter, args.inputmean, args.inputstd)
    print('objects detected', results)
    print('labels detected', labels)
    print('rectangles', rects)
    interpreter, possible_labels = init_tf2(args.species_model, args.numthreads, args.species_labels)
    result, label = set_label(img, possible_labels, interpreter, args.inputmean, args.inputstd)
    print('final result', result)
    print('final label', label)


# load label file for obj detection or classification model
def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


# initialize tensor flow model
def init_tf2(model_file, num_threads, label_file):
    possible_labels = np.asarray(load_labels(label_file))  # load label file and convert to list
    try:  # load tensorflow lite on rasp pi
        interpreter = tflite.Interpreter(model_file, num_threads)
    except:  # load full tensor for desktop dev
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
    interpreter.invoke()

    if floating_model is False:  # tensor lite obj detection prebuilt model
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
    # else full tensor model, add code here
    # print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
    return confidences, confidence_labels, confidence_rects  # confidence and best label


# input image and return best result and label
def set_label(img, labels, interpreter, input_mean, input_std):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    floating_model, input_data = convert_cvframe_to_ts(img, input_details, input_mean, input_std)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

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
        cv2.imwrite("debugimg.jpg", img)

    cresult = cresult / 100  # needed for automl or google coral.ai model
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
    colorcount = [''] * 4
    colors = np.array(['Red', 'Blue', 'Yellow', 'Gray'])
    boundaries = [
        ([17, 15, 100], [50, 56, 200]),  # Red
        ([86, 31, 4], [220, 88, 50]),  # Blue
        ([25, 146, 190], [62, 174, 250]),  # Yellow
        ([103, 86, 65], [145, 133, 128])  # Gray
    ]

    # grab background color for initial mask
    # bkgcolor = int(img[0, 0])  # bgr value in upper left corner
    # bkgmask = cv2.inRange(img, bkgcolor, ...)

    # loop over the boundaries
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")  # create NumPy arrays from the boundaries
        upper = np.array(upper, dtype="uint8")
        # find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(img, lower, upper)
        maskimg = cv2.bitwise_and(img, img, mask=mask)
        colorcount[i] = np.count_nonzero(maskimg, axis=None)  # count non-zero values in BGR pixels
        i += 1

    cindex = np.where(colorcount == np.amax(colorcount))  # find color with highest count
    color = str(colors[cindex])
    return color


# add bounding box and label to an image
def add_box_and_label(img, img_label, startX, startY, endX, endY, colors, coloroffset):
    cv2.rectangle(img, (startX, startY), (endX, endY), colors[coloroffset], 2)
    y = startY - 15 if startY - 15 > 15 else startY + 15  # adjust label loc if too low
    cv2.putText(img, img_label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[coloroffset], 2)
    return img


# test function
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # ap.add_argument('-i', '--image', default='/home/pi/birdclass/test2.jpg',help='image to be classified')
    ap.add_argument('-i', '--image', default='/home/pi/birdclass/cardinal.jpg', help='image to be classified')

    # object detection model setup
    ap.add_argument('-om', "--obj_det_model",
                    default='/home/pi/birdclass/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite')
    ap.add_argument('-p', '--obj_det_labels',
                    default='/home/pi/birdclass/lite-model_ssd_mobilenet_v1_1_metadata_2_labelmap.txt')

    # species model setup
    ap.add_argument('-m', '--species_model',
                    default='/home/pi/birdclass/coral.ai.mobilenet_v2_1.0_224_inat_bird_quant.tflite',
                    help='.tflite model to be executed')
    ap.add_argument('-l', '--species_labels',
                    default='/home/pi/birdclass/coral.ai.inat_bird_labels.txt',
                    help='name of file containing labels')

    # tensor flow input arguements
    ap.add_argument('--inputmean', default=127.5, type=float, help='Tensor input_mean')
    ap.add_argument('--inputstd', default=127.5, type=float, help='Tensor input standard deviation')
    ap.add_argument('--numthreads', default=None, type=int, help='Tensor number of threads')

    # confidence settings for object detection and species bconfidence
    ap.add_argument('-bc', '--bconfidence', type=float, default=0.80)
    ap.add_argument('-sc', '--sconfidence', type=float, default=0.95)

    arguments = ap.parse_args()

    main(arguments)
