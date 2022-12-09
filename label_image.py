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

import numpy as np
import random
from PIL import ImageDraw as PILImageDraw
import image_proc
import math

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
    def __init__(self, homedir='/home/pi/PycharmProjects/birdclassifier/',
                 object_model='lite-model_ssd_mobilenet_v1_1_metadata_2.tflite',
                 object_model_labels='lite-model_ssd_mobilenet_v1_1_metadata_2_labelmap.txt',
                 classifier_model='coral.ai.mobilenet_v2_1.0_224_inat_bird_quant.tflite',
                 classifier_labels='coral.ai.inat_bird_labels.txt',
                 classifier_thresholds='coral.ai.inat_bird_threshold.csv',
                 detect_object_min_confidence=.9, screenheight=480,
                 screenwidth=640, contrast_chg=1.0, color_chg=1.0, brightness_chg=1.0, sharpness_chg=1.0,
                 min_img_percent=10.0, min_area=28000, target_object='bird',
                 classify_object_min_confidence=.8, output_class=None, verbose=False):
        self.detector_file = homedir + object_model  # object model
        self.detector_labels_file = homedir + object_model_labels  # obj model label
        self.target_objects = target_object
        self.target_object_found = False
        self.classifier_file = homedir + classifier_model  # classifier model
        self.labels = classifier_labels
        self.thresholds = classifier_thresholds
        self.classifier_labels_file = homedir + classifier_labels
        self.classifier_thresholds_file = homedir + classifier_thresholds
        self.classifier_thresholds = np.genfromtxt(self.classifier_thresholds_file, delimiter=',')
        self.detector, self.obj_detector_possible_labels = self.init_tf2(self.detector_file, self.detector_labels_file)
        self.classifier, self.classifier_possible_labels = self.init_tf2(self.classifier_file,
                                                                         self.classifier_labels_file)
        self.input_mean = 127.5  # recommended default
        self.input_std = 127.5  # recommended default
        self.detect_obj_min_confidence = detect_object_min_confidence
        self.classify_object_min_confidence = classify_object_min_confidence
        self.obj_confidence = 0
        self.min_area = min_area
        self.detected_confidences = []
        self.detected_labels = []
        self.detected_rects = []
        self.classified_confidences = []
        self.classified_labels = []
        self.classified_rects = []
        self.classified_rects_area = []
        self.last_known_classified_confidences = []
        self.last_known_classified_labels = []
        self.last_known_classified_rects = []
        self.last_known_classified_rects_area = []
        self.colors = np.random.uniform(0, 255, size=(11, 3)).astype(int)  # random colors for bounding boxes
        self.color_index = self.pick_a_color()  # set initial color to use for bounding boxes
        self.text_color_index = self.pick_a_color()  # set initial color to use for text

        # set image adjustment parameters
        self.contrast_chg = contrast_chg
        self.brightness_chg = brightness_chg
        self.color_chg = color_chg
        self.sharpness_chg = sharpness_chg
        self.min_img_percent = min_img_percent
        self.screenwidth = screenwidth
        self.screenheight = screenheight
        self.screen_sq_pixels = screenwidth * screenheight
        self.img = np.zeros((screenheight, screenwidth, 3), dtype=np.uint8)
        self.output_class = output_class
        self.output_function = output_class.message if output_class is not None else print
        self.verbose = verbose

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

    # set working image, add equalized img, detect objects, return boolean if detected objects in
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
            for index, self.obj_confidence in enumerate(det_confidences[0]):
                labelidx = int(det_labels_index[0][index])  # get result label index for labels;
                label = self.obj_detector_possible_labels[labelidx]  # grab text from possible labels
                # if self.verbose and self.obj_confidence >= .5:
                #     self.output_function(f'bird detection confidence: {self.obj_confidence}')
                if self.obj_confidence >= self.detect_obj_min_confidence and \
                        label in self.target_objects:
                    self.target_object_found = True
                    self.detected_confidences.append(self.obj_confidence)
                    self.detected_labels.append(label)
                    self.detected_rects.append(det_rects[0][index])
        return self.target_object_found

    # make classifications using img and detect results
    # compare img and equalized images
    # use self.detected_confidences,detected_labels, detected_rects lists
    # stash prior classification for later processing if current frame does not find a bird
    def classify(self, img, use_confidence_threshold=True):
        self.classified_rects = []
        self.classified_rects_area = []
        self.classified_confidences = []
        self.classified_labels = []
        prior_rect = (0, 0, 0, 0)
        for i, det_confidence in enumerate(self.detected_confidences):  # loop thru detected target objects
            (startX, startY, endX, endY) = self.scale_rect(img, self.detected_rects[i])  # set x,y bounding box
            rect = (startX, startY, endX, endY)
            rect_percent_scr = ((endX - startX) * (endY - startY)) / self.screen_sq_pixels * 100  # % of screen of img
            crop_img = img.crop((startX, startY, endX, endY))  # extract image for better classification
            equalizedimg = image_proc.enhance(img, brightness=self.brightness_chg, contrast=self.contrast_chg,
                                              color=self.color_chg, sharpness=self.sharpness_chg)
            crop_equalizedimg = equalizedimg.crop((startX, startY, endX, endY))
            classify_conf, classify_label = self.classify_obj(crop_img, rect, use_confidence_threshold,
                                                              rect_percent_scr)
            classify_conf_equalized, classify_label_equalized = self.classify_obj(crop_equalizedimg, rect,
                                                                                  use_confidence_threshold,
                                                                                  rect_percent_scr)
            # take the best result between img and enhanced img
            classify_label = classify_label if classify_conf >= classify_conf_equalized else classify_label_equalized
            classify_conf = classify_conf if classify_conf >= classify_conf_equalized else classify_conf_equalized
            if classify_conf != 0:
                self.output_function(f'match returned: confidence {classify_conf:.3f}, {classify_label},',
                                     msg_type='match')

            _overlap_perc = image_proc.overlap_area(prior_rect, rect)  # compare current rect and prior rect
            prior_rect = rect  # set prior rect to current rect
            # print('overlap percent', overlap_perc)
            # record bird classification and location if there is a label
            if len(classify_label.strip()) > 0:
                self.classified_labels.append(classify_label)
                self.classified_confidences.append(classify_conf)
                self.classified_rects.append(rect)
                self.classified_rects_area.append(rect_percent_scr)
        if max(self.classified_confidences, default=0) == 0:  # if empty list or zero
            max_confidence = 0
        else:  # set last known to current species if confident
            max_confidence = max(self.classified_confidences)
            self.last_known_classified_rects = self.classified_rects
            self.last_known_classified_rects_area = self.classified_rects_area
            self.last_known_classified_labels = self.classified_labels
            self.last_known_classified_confidences = self.classified_confidences
        return max_confidence

    # input image and return best result and label
    # the function will sort the results and compare the confidence to the confidence for that label (species)
    # if the ML models confidence is higer than the treshold for that lable (species) it will stop searching and
    # return that best result
    def classify_obj(self, img, rect, use_confidence_threshold=True, screen_percent=100.00):
        input_details = self.classifier.get_input_details()
        floating_model, input_data = self.convert_img_to_tf(img, input_details)
        self.classifier.set_tensor(input_details[0]['index'], input_data)
        self.classifier.invoke()  # invoke classification
        output_details = self.classifier.get_output_details()[0]
        output = np.squeeze(self.classifier.get_tensor(output_details['index']))
        # If the model is quantized (tflite uint8 data), then dequantize the results
        if output_details['dtype'] == np.uint8:
            scale, zero_point = output_details['quantization']
            output = scale * (output - zero_point) * 10  # scale factor to adjust results
        cindex = np.argpartition(output, -10)[-10:]
        # loop thru top N results to find best match; highest score align with matching species threshold
        maxcresult = float(0)
        maxlresult = ''
        for lindex in cindex:
            lresult = str(self.classifier_possible_labels[lindex]).strip()  # grab label,push to string instead of tuple
            cresult = float(output[lindex]) if float(output[lindex]) > 0 else 0
            cresult = cresult - math.floor(cresult) if cresult > 1 else cresult  # ignore whole numbers, keep decimals
            if self.check_threshold(cresult, lindex, use_confidence_threshold, screen_percent):
                if cresult > maxcresult:  # if this above threshold and is a better confidence result store it
                    maxcresult = cresult
                    maxlresult = lresult
        # if maxcresult != 0:
        #     self.output_function(f'match returned: confidence {maxcresult:.3f}, {maxlresult}')
        return maxcresult, maxlresult  # highest confidence with best match

    # takes a PIL image type and converts it to np array for tensor
    # resize for classification model
    def convert_img_to_tf(self, pil_img, input_details):
        # check the type of the input tensor
        floating_model = input_details[0]['dtype'] == np.float32
        # NxHxWxC, H:1, W:2
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        reshape_image = pil_img.resize((width, height))
        # try:
        #     if tfliteb is True:  # make sure this is running on the pi and not testing on the PC
        #         reshape_image = tflite.image.resize_with_pad(image=pil_img, target_width=width, target_height=height)
        #     else:
        #         reshape_image = tf.image.resize_with_pad(image=pil_img, target_width=width, target_height=height)
        # except Exception as e:
        #     print(e)
        #     pass
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
        return x_min, y_min, x_max, y_max

    # add bounding box and label to an image
    # we may have a rect with no species and a zero confidence, in that case use the last known label and confidence
    def add_boxes_and_labels(self, img, use_last_known=False):
        if use_last_known and round(max(self.classified_confidences, default=0), 2) == 0:
            # print('using last known', self.last_known_classified_confidences, self.last_known_classified_labels)
            classified_rects = self.last_known_classified_rects
            classified_rects_area = self.last_known_classified_rects_area
            classified_labels = self.last_known_classified_labels
            classified_confidences = self.last_known_classified_confidences
        else:
            classified_rects = self.classified_rects
            classified_rects_area = self.classified_rects_area
            classified_labels = self.classified_labels
            classified_confidences = self.classified_confidences
        for i, rect in enumerate(classified_rects):
            (start_x, start_y, end_x, end_y) = rect
            draw = PILImageDraw.Draw(img)
            font = draw.getfont()
            try:  # add text to top and bottom of image, make box slightly large and put text on top and bottom
                # font = font, fill = self.text_color if color font is desired
                rect_area = classified_rects_area[i] if len(classified_rects_area) > 0 \
                    else (rect[2] - rect[0]) * (rect[3] - rect[1])  # if area is empty calculate from rect
                draw.text((start_x, start_y-50), self.label_text(classified_labels[i], classified_confidences[i],
                                                                 rect_area), font=font, fill='white')
                draw.text((start_x, end_y+50), self.label_text(classified_labels[i], classified_confidences[i],
                                                               rect_area), font=font, fill='white')
                draw.line([(start_x-25, start_y-25), (start_x-25, end_y+25), (start_x-25, end_y+25),
                           (end_x+25, end_y+25), (end_x+25, end_y+25),
                           (end_x+25, start_y-25), (end_x+25, start_y-25), (start_x-25, start_y-25)],
                          fill=self.get_next_color(from_index=i), width=2)
            except Exception as e:
                print('tried drawing text on the image of ith rectangle with rect:', i)
                print(e)
                print(classified_labels, classified_confidences, classified_rects, classified_rects_area)
        return img

    # used to get a temp copy of data if labeling is delayed
    def get_obj_data(self):
        return self.classified_rects, self.classified_labels, self.classified_confidences

    def set_ojb_data(self, classified_rects, classified_labels, classified_confidences):
        self.classified_rects = classified_rects
        self.classified_labels = classified_labels
        self.classified_confidences = classified_confidences
        return

    # pick random color for stream of frames
    def pick_a_color(self):
        # return tuple(self.colors[random.randint(0, (len(self.colors)) - 1)])
        return random.randint(0, (len(self.colors)) - 1)

    def set_colors(self):
        self.color_index = self.pick_a_color()
        self.text_color_index = self.pick_a_color()
        return

    def get_next_color(self, from_index=0):
        return tuple(self.colors[(self.color_index + from_index) % (len(self.colors) - 1)])

    # func checks threshold by each label passed as a nparray with text in col 0 and threshold in col 1
    # object cannot be -1 (not present in geo location), confidence cannot be 0,must be equal or exceed minimum score
    # img must take up the specified min % of the image or it is tossed out
    # cresult is a decimal % 0 - 1; lindex is % * 10 (no decimals) must div by 1000 to get same scale
    def check_threshold(self, cresult, lindex, use_confidence_threshold, screen_percent):
        # grab default if species has 0 confidence, else use species specific score
        label_threshold = self.classify_object_min_confidence * 1000 \
            if self.classifier_thresholds[int(lindex)][1] == 0 \
            else self.classifier_thresholds[int(lindex)][1]
        # push to zero if use threshold boolean is false, this automatically puts any confidence over the threshold
        label_threshold = label_threshold if (use_confidence_threshold or label_threshold == -1) else 0
        try:  # handle typos in threshold file or unexpected results from model
            label_threshold = float(label_threshold)
            cresult = float(cresult)
        except Exception as e:
            print(e)
            print(label_threshold)
            print(cresult)
            label_threshold = 0
            cresult = 0
        return(float(label_threshold) != -1 and screen_percent >= self.min_img_percent and
               cresult > 0 and cresult >= float(label_threshold) / 1000)

    # set label for box in image use short species name instead of scientific name
    def label_text(self, label, confidence, screen_percent):
        sname = str(label)  # make sure label is considered a string
        start = sname.find('(') + 1  # find start of common name, move one character to drop (
        end = sname.find(')')
        cname = sname[start:end] if start >= 0 and end >= 0 else sname
        common_name = f'{cname} {confidence * 100:.2f}%, frame:{float(screen_percent):.2f}%'
        return common_name


# def main(args):
#     img = PILImage.open(args.image)  # load image
#     birds = DetectClassify(args.homedir, default_confidence=.9)  # setup obj
#     birds.detect(img)  # run object detection
#
#     print('objects detected', birds.detected_confidences)
#     print('labels detected', birds.detected_labels)
#     print('rectangles', birds.detected_rects)
#
#     birds.classify(birds.img)  # classify species
#     print(birds.classified_labels)
#     label = birds.classified_labels[0]
#     if len(label) == 0:
#         label = "no classification text"
#     print(label)
#     img = birds.add_boxes_and_labels(img)
#     img.save('imgtest.jpg')
#     img.show()


# test function
# if __name__ == '__main__':
#     ap = argparse.ArgumentParser()
#     ap.add_argument('-i', '--image', default='/home/pi/birdclass/test2.jpg', help='cardinal')
#     ap.add_argument('-dir', '--homedir', default='c:/Users/jimma/PycharmProjects/birdclassifier/', help='loc files')
#
#     arguments = ap.parse_args()
#
#     main(arguments)
