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
except Exception as e:
    print(e)
    print('picamera import fails on windows')
    pass

# attempt to load tensor flow lite,will fail if not raspberry pi, switch to full tensorflow for windows
try:
    import tflite_runtime.interpreter as tflite  # for pi4 with install wheel above
    tfliteb = True
except Exception as e:
    print(e)
    print('load TF2 for testing')
    import tensorflow as tf  # TF2 for desktop testing
    tfliteb = False


class DetectClassify:
    def __init__(self, homedir='/home/pi/PycharmProjects/birdclassifier/', default_confidence=.98, screenheight=640,
                 screenwidth=480, contrast_chg=1.0, color_chg=1.0, brightness_chg=1.0, sharpness_chg=1.0,
                 mismatch_penalty=0.3, overlap_perc_tolerance=0.7):
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
        self.classify_default_confidence = default_confidence
        self.classify_mismatch_reduction = mismatch_penalty
        self.detected_confidences = []
        self.detected_labels = []
        self.detected_rects = []
        self.classified_confidences = []
        self.classified_labels = []
        self.classified_rects = []
        self.last_known_classified_confidences = []
        self.last_known_classified_labels = []
        self.last_known_classified_rects = []
        self.colors = np.random.uniform(0, 255, size=(11, 3)).astype(int)  # random colors for bounding boxes
        self.color = self.pick_a_color()  # set initial color to use for bounding boxes

        # set image adjustment parameters
        self.contrast_chg = contrast_chg
        self.brightness_chg = brightness_chg
        self.color_chg = color_chg
        self.sharpness_chg = sharpness_chg
        self.overlap_perc_tolerance = overlap_perc_tolerance
        self.img = np.zeros((screenheight, screenwidth, 3), dtype=np.uint8)

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
    # stash prior classification for later processing if current frame does not find a bird
    def classify(self, img):
        self.classified_rects = []
        self.classified_confidences = []
        self.classified_labels = []
        # prior_rect = (0, 0, 0, 0)
        # overlap_perc = 0.0
        for i, det_confidence in enumerate(self.detected_confidences):  # loop thru detected target objects
            (startX, startY, endX, endY) = self.scale_rect(img, self.detected_rects[i])  # set x,y bounding box
            rect = (startX, startY, endX, endY)
            crop_img = img.crop((startX, startY, endX, endY))  # extract image for better classification
            equalizedimg = image_proc.enhance(img, brightness=self.brightness_chg, contrast=self.contrast_chg,
                                              color=self.color_chg, sharpness=self.sharpness_chg)
            crop_equalizedimg = equalizedimg.crop((startX, startY, endX, endY))
            classify_conf, classify_label = self.classify_obj(crop_img)
            classify_conf_equalized, classify_label_equalized = self.classify_obj(crop_equalizedimg)
            if classify_label != classify_label_equalized:  # labels should match if pic quality is good
                if classify_conf < classify_conf_equalized:
                    classify_conf = classify_conf_equalized
                    classify_label = classify_label_equalized
                classify_conf -= self.classify_mismatch_reduction  # reduce confidence on mismatch
            else:  # increase confidence on match, use classify_label already set above
                classify_conf = \
                    classify_conf + classify_conf_equalized if classify_conf + classify_conf_equalized <= 1 else 1

            # detect overlapping rectangles/same bird and skip it
            # overlap_perc = image_proc.overlap_area(prior_rect, rect)  # compare current rect and prior rect
            # prior_rect = rect  # set prior rect to current rect
            # if overlap_perc > self.overlap_perc_tolerance:  # 0.0 in first loop, if 80% overlap skip bird
            #     classify_conf = 0
            #     classify_label = ""

            self.classified_labels.append(classify_label)
            classify_conf = classify_conf if classify_conf > 0 else 0  # check for negative conf
            self.classified_confidences.append(classify_conf)
            self.classified_rects.append(rect)
        if len(self.classified_confidences) == 0:  # check for an empty list
            max_confidence = 0
        elif round(max(self.classified_confidences), 3) == 0:  # if not empty check zero, don't combine with above if
            max_confidence = 0
        else:
            max_confidence = max(self.classified_confidences)
            self.last_known_classified_rects = self.classified_rects
            self.last_known_classified_labels = self.classified_labels
            self.last_known_classified_confidences = self.classified_confidences
        return max_confidence

    # input image and return best result and label
    # the function will sort the results and compare the confidence to the confidence for that label (species)
    # if the ML models confidence is higer than the treshold for that lable (species) it will stop searching and
    # return that best result
    def classify_obj(self, img):
        input_details = self.classifier.get_input_details()
        # output_details = self.classifier.get_output_details()
        floating_model, input_data = self.convert_img_to_tf(img, input_details)
        self.classifier.set_tensor(input_details[0]['index'], input_data)
        self.classifier.invoke()  # invoke classification
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

    # create bounding coordinates for rectangle and text position from image size
    def scale_rect(self, img, box):
        img_width, img_height = img.size
        y_min = int(max(1, (box[0] * img_height)))
        x_min = int(max(1, (box[1] * img_width)))
        y_max = int(min(img_height, (box[2] * img_height)))
        x_max = int(min(img_width, (box[3] * img_width)))
        return (x_min, y_min, x_max, y_max)

    # add bounding box and label to an image
    # we may have a rect with no species and a zero confidence, in that case use the last known label and confidence
    def add_boxes_and_labels(self, img, use_last_known=False):
        if use_last_known and len(self.classified_confidences) == 0:
            classified_rects = self.last_known_classified_rects
            classified_labels = self.last_known_classified_labels
            classified_confidences = self.last_known_classified_confidences
        else:
            classified_rects = self.classified_rects
            classified_labels = self.classified_labels
            classified_confidences = self.classified_confidences

        for i, rect in enumerate(classified_rects):
            try:
                (start_x, start_y, end_x, end_y) = rect
                text_x = start_x
                text_y = start_y
                start_x += -25
                start_y += -25
                end_x += 25
                end_y += 25
            except TypeError:
                print('TypeError in add boxes and labels', rect)
                return
            draw = PILImageDraw.Draw(img)
            font = draw.getfont()
            draw.text((text_x, text_y), self.label_text(classified_labels[i], classified_confidences[i]),
                      font=font, color=self.color)
            draw.line([(start_x, start_y), (start_x, end_y), (start_x, end_y), (end_x, end_y),
                       (end_x, end_y), (end_x, start_y), (end_x, start_y), (start_x, start_y)],
                      fill=self.color, width=2)
        return img

    # pick random color for stream of frames
    def pick_a_color(self):
        return tuple(self.colors[random.randint(0, (len(self.colors)) - 1)])

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

    # set label for box in image use short species name instead of scientific name
    def label_text(self, label, confidence):
        sname = str(label)  # make sure label is considered a string
        start = sname.find('(') + 1  # find start of common name, move one character to drop (
        end = sname.find(')')
        cname = sname[start:end] if start >= 0 and end >= 0 else sname
        common_name = f'{cname} {confidence * 100:.1f}%'
        return common_name


def main(args):
    img = PILImage.open(args.image)  # load image
    birds = DetectClassify(args.homedir, default_confidence=.9)  # setup obj
    birds.detect(img)  # run object detection

    print('objects detected', birds.detected_confidences)
    print('labels detected', birds.detected_labels)
    print('rectangles', birds.detected_rects)

    birds.classify(birds.img)  # classify species
    print(birds.classified_labels)
    label = birds.classified_labels[0]
    if len(label) == 0:
        label = "no classification text"
    print(label)
    img = birds.add_boxes_and_labels(img)
    img.save('imgtest.jpg')
    img.show()


# test function
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', default='/home/pi/birdclass/test2.jpg', help='cardinal')
    ap.add_argument('-dir', '--homedir', default='c:/Users/jimma/PycharmProjects/birdclassifier/', help='loc files')

    arguments = ap.parse_args()

    main(arguments)
