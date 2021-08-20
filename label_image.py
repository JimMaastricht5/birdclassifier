# MIT License
#
# 2021 Jim Maastricht
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# JimMaastricht5@gmail.com
# refactored for object detection and object classification
# blended tensor and tensor flow lite capabilities
# added scale rect results from obj detection to apply to full image
# added code for detailed object detection and for general classification
# ==============================================================================
# PY4: pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import math
import random
from PIL import Image as PILImage
# from PIL import ImageFont as PILImageFont
from PIL import ImageDraw as PILImageDraw
import image_proc

# attempt to load picamera, fails on windows
try:
    import picamera
except:
    pass

# attempt to load tensor flow lite,will fail if not raspberry pi, switch to full tensorflow for windows
try:
    import tflite_runtime.interpreter as tflite  # for pi4 with install wheel above
    tfliteb = True
except:
    import tensorflow as tf  # TF2 for desktop testing
    tfliteb = False


class DetectClassify:
    def __init__(self, homedir='/home/pi/PycharmProjects/pyface2/', default_confidece=.98):
        self.detector_file = homedir + 'lite-model_ssd_mobilenet_v1_1_metadata_2.tflite'
        self.detector_labels_file = homedir + 'lite-model_ssd_mobilenet_v1_1_metadata_2_labelmap.txt'
        self.target_objects = ['bird']
        self.target_object_found = False
        self.classifier_file = homedir + 'coral.ai.mobilenet_v2_1.0_224_inat_bird_quant.tflite'
        self.classifier_labels_file = homedir + 'coral.ai.inat_bird_labels.txt'
        self.classifier_thresholds_file = homedir + 'coral.ai.inat_bird_threshold.csv'
        self.classifier_thresholds = np.genfromtxt(self.classifier_thresholds_file, delimiter=',')
        self.detector, self.obj_detector_possible_labels = self.init_tf2(self.detector_file, self.detector_labels_file)
        self.classifier, self.classifier_possible_labels = self.init_tf2(self.classifier_file,
                                                                         self.classifier_labels_file)
        self.input_mean = 127.5  # recommended default
        self.input_std = 127.5  # recommended default
        self.detect_obj_min_confidence = .6
        self.classify_min_confidence = .7
        self.classify_default_confidence = default_confidece
        self.classify_mismatch_reduction = .30
        self.detected_confidences = []
        self.detected_labels = []
        self.detected_rects = []
        self.classified_confidences = []
        self.classified_labels = []
        self.classified_rects = []
        self.colors = np.random.uniform(0, 255, size=(11, 3)).astype(int)  # random colors for bounding boxes
        self.img = np.zeros((640, 480, 3), dtype=np.uint8)
        self.equalizedimg = np.zeros((640, 480, 3), dtype=np.uint8)
        try:
            self.camera = picamera.PiCamera()
            self.camera.resolution(640, 480)
            self.camera.framerate = 30
        except:
            pass

    # initialize tensor flow model
    def init_tf2(self, model_file, label_file_name):
        possible_labels = np.asarray(self.load_labels(label_file_name))  # load label file and convert to list
        try:  # load tensorflow lite on rasp pi
            interpreter = tflite.Interpreter(model_file, None)
        except NameError:  # load full tensor for desktop dev
            interpreter = tf.lite.Interpreter(model_file, None)
        interpreter.allocate_tensors()
        return interpreter, possible_labels

    # load label file for obj detection or classification model
    def load_labels(self, filename):
        with open(filename, 'r') as f:
            return [line.strip() for line in f.readlines()]

    # set working image, add equalized img, detect objects, return boolean if detected objecets in
    # target list.  Create color equalized version of img
    # fill detected_confidences, detected_labels, and detected_rects if in target object list
    def detect(self, img):
        self.img = img.copy()
        self.equalizedimg = image_proc.equalize_color(img)  # balance histogram of color intensity for all frames
        self.detected_confidences = []
        self.detected_labels = []  # possible object labels
        self.detected_rects = []
        self.target_object_found = False

        input_details = self.detector.get_input_details()
        output_details = self.detector.get_output_details()
        floating_model, input_data = self.convert_img_to_tf(self.img, input_details)
        self.detector.set_tensor(input_details[0]['index'], input_data)
        self.detector.invoke()

        if floating_model is False:  # tensor lite obj detection prebuilt model
            det_rects = self.detector.get_tensor(output_details[0]['index'])
            det_labels_index = self.detector.get_tensor(output_details[1]['index'])  # label array for each result
            det_confidences = self.detector.get_tensor(output_details[2]['index'])
            for index, det_confidence in enumerate(det_confidences[0]):
                labelidx = int(det_labels_index[0][index])  # get result label index for labels;
                label = self.obj_detector_possible_labels[labelidx]  # grab text from possible labels
                if det_confidence >= self.detect_obj_min_confidence and \
                        label in self.target_objects:
                    self.target_object_found = True
                    self.detected_confidences.append(det_confidence)
                    self.detected_labels.append(label)
                    self.detected_rects.append(det_rects[0][index])
        return self.target_object_found

        # make classifications using img and detect results
        # compare img and equalized images
        # use self.detected_confidences,detected_labels, detected_rects lists

    def classify(self):
        self.classified_rects = []
        self.classified_confidences = []
        self.classified_labels = []
        for i, det_confidence in enumerate(self.detected_confidences):  # loop thru detected target objects
            (startX, startY, endX, endY) = self.scale_rect(self.img, self.detected_rects[i])  # set x,y bounding box
            rect = (startX, startY, endX, endY)
            crop_img = self.img.crop((startX, startY, endX, endY))  # extract image for better classification
            crop_equalizedimg = self.equalizedimg.crop((startX, startY, endX, endY))
            classify_conf, classify_label = self.classify_obj(crop_img)
            classify_conf_equalized, classify_label_equalized = self.classify_obj(crop_equalizedimg)
            if classify_label != classify_label_equalized:  # predictions should match if pic quality is good
                # pick the result with highest confidence and reduce the confidence of the highest
                if classify_conf >= classify_conf_equalized:
                    pass
                else:
                    classify_conf = classify_conf_equalized
                    classify_label = classify_label_equalized
                classify_conf -= self.classify_mismatch_reduction  # reduce confidence on mismatch
            else:  # increase confidence on match
                classify_conf += classify_conf_equalized
                if classify_conf > 1:
                    classify_conf = 1

            self.classified_labels.append(classify_label)
            self.classified_confidences.append(classify_conf)
            self.classified_rects.append(rect)
        return

    # input image and return best result and label
    # the function will sort the results and compare the confidence to the confidence for that label (species)
    # if the ML models confidence is higer than the treshold for that lable (species) it will stop searching and
    # return that best result
    def classify_obj(self, img):
        input_details = self.classifier.get_input_details()
        # output_details = self.classifier.get_output_details()
        floating_model, input_data = self.convert_img_to_tf(img, input_details)
        self.classifier.set_tensor(input_details[0]['index'], input_data)

        self.classifier.invoke()
        output_details = self.classifier.get_output_details()[0]
        output = np.squeeze(self.classifier.get_tensor(output_details['index']))
        # If the model is quantized (tflite uint8 data), then dequantize the results
        if output_details['dtype'] == np.uint8:
            scale, zero_point = output_details['quantization']
            output = scale * (output - zero_point) * 10  # added * 10 scale factor to try and get numbers right

        cindex = np.argpartition(output, -10)[-10:]
        # loop thru top N results to find best match; highest score align with matching species threshold
        maxcresult = float(0)
        maxlresult = ''
        for lindex in cindex:
            lresult = str(self.classifier_possible_labels[lindex])  # grab label,push to string instead of tuple
            cresult = float(output[lindex])  # grab predicted confidence score
            if cresult != 0:
                # print(f'     {check_threshold(cresult, lindex, label_thresholds)} match, confidence:{str(cresult)}' +
                #         f', threshold:{label_thresholds[lindex][1]}, {str(labels[lindex])}.')
                if cresult > 1:  # still don't have scaling working 100% if the result is more than 100% adjust
                    if tfliteb:
                        cresult -= math.floor(cresult)
                    else:
                        cresult = cresult / 10
                if self.check_threshold(cresult, lindex):  # comp confidence>=threshold by label
                    if cresult > maxcresult:  # if this above threshold and is a better confidence result store it
                        maxcresult = cresult
                        maxlresult = lresult
        if maxcresult != 0:
            print(f'match returned: confidence {maxcresult}, {maxlresult}')
        return maxcresult, maxlresult  # highest confidence with best match

    # takes a PIL image type and converts it to np array for tensor
    # resize for classification model
    def convert_img_to_tf(self, pil_img, input_details):
        # check the type of the input tensor
        floating_model = input_details[0]['dtype'] == np.float32
        # NxHxWxC, H:1, W:2
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        reshape_image = pil_img.resize((width, height), PILImageDraw.Image.LANCZOS)
        image_np = image_proc.convert(reshape_image, 'np')
        image_np_expanded = np.expand_dims(image_np, axis=0)

        if floating_model:
            input_data = image_np_expanded.astype('float32')
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std
        else:
            input_data = image_np_expanded.astype('uint8')
        return floating_model, input_data

    def scale_rect(self, img, box):
        img_width, img_height = img.size
        y_min = int(max(1, (box[0] * img_height)))
        x_min = int(max(1, (box[1] * img_width)))
        y_max = int(min(img_height, (box[2] * img_height)))
        x_max = int(min(img_width, (box[3] * img_width)))
        return (x_min, y_min, x_max, y_max)

    # add bounding box and label to an image
    def add_boxes_and_labels(self, img, label, rects):
        transparent_fill = (0, 0, 0, 0)  # white with transparent alpha
        for i, rect in enumerate(rects):
            try:
                (startX, startY, endX, endY) = rect
            except TypeError:
                return
            color = tuple(self.colors[random.randint(0, (len(self.colors)) - 1)])
            print(color)

            draw = PILImageDraw.Draw(img)
            font = draw.getfont()
            draw.text((startX, startY), label, font=font, color=color, fill=transparent_fill)
            # draw.rectangle([(startX, startY), (endX, endY)],  outline=color, width=1, fill=transparent_fill)
        return img

    # func checks threshold by each label passed as a nparray with text in col 0 and threshold in col 1
    # species cannot be -1 (not present in geo location), cannot be 0, and must be equal or exceed minimum score
    # cresult is a decimal % 0 - 1; lindex is % * 10 (no decimals) must div by 1000 to get same scale
    def check_threshold(self, cresult, lindex):
        if self.classifier_thresholds[int(lindex)][1] == 0:
            label_threshold = self.classify_default_confidence * 1000
        else:
            label_threshold = self.classifier_thresholds[int(lindex)][1]
        return(int(label_threshold) != -1 and cresult > 0 and
               cresult >= float(label_threshold) / 1000)


def main(args):
    img = PILImage.open(args.image)  # load image
    birds = DetectClassify(args.homedir, default_confidece=.9)  # setup obj
    birds.detect(img)  # run object detection

    print('objects detected', birds.detected_confidences)
    print('labels detected', birds.detected_labels)
    print('rectangles', birds.detected_rects)

    birds.classify()  # classify species
    print(birds.classified_labels)
    label = birds.classified_labels[0]
    if len(label) == 0:
        label = "no classification text"
    print(label)
    img = birds.add_boxes_and_labels(img, label, birds.classified_rects)
    img.show()


# test function
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', default='/home/pi/birdclass/test2.jpg', help='cardinal')
    ap.add_argument('-dir', '--homedir', default='c:/Users/jimma/PycharmProjects/birdclassifier/', help='loc files')

    arguments = ap.parse_args()

    main(arguments)
