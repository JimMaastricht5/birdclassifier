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
# Module performs object detection (bird? and box boundary) as well as
# classification (what species)
# supports tensor (windows) and tensor flow lite (rasp pi) capabilities
#
# Note on pi if problems with TF Lite install use something similar to below, most recent setup it was
# not required (OS Bookworm)
# pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
from typing import Union
import numpy as np
import random
from PIL import ImageDraw as PILImageDraw
from PIL import Image
import image_proc
import math
import static_functions
import csv


# attempt to load tensor flow lite,will fail if not raspberry pi, switch to full tensorflow for windows
try:
    import tflite_runtime.interpreter as tflite  # for pi4
except Exception as tf_e:
    print(tf_e)
    print('load full TF for testing')
    import tensorflow as tf  # TF2 for desktop testing


class DetectClassify:
    """
    Class will detect the target object, if found it will create the bounding box and
    pass it on to the classifier to for species identification.  The class was tested with
    birds, but is designed to be used generically with a few small changes
    """
    def __init__(self, homedir: str = '/home/pi/birdclassifier/',
                 object_model: str = 'lite-model_ssd_mobilenet_v1_1_metadata_2.tflite',
                 object_model_labels: str = 'lite-model_ssd_mobilenet_v1_1_metadata_2_labelmap.txt',
                 classifier_model: str = 'coral.ai.mobilenet_v2_1.0_224_inat_bird_quant.tflite',
                 classifier_labels: str = 'coral.ai.inat_bird_labels.txt',
                 classifier_thresholds: str = 'USA_WI_coral.ai.inat_bird_threshold.csv',
                 detect_object_min_confidence: float = .6, screenwidth: int = 480, screenheight: int = 640,
                 contrast_chg: float = 1.0, color_chg: float = 1.0,
                 brightness_chg: float = 1.0, sharpness_chg: float = 1.0,
                 min_img_percent: float = 10.0, target_object: Union[list, str] = 'bird',
                 classify_object_min_confidence: float = .9,
                 debug: bool = False,
                 output_class=None) -> None:
        """
        set up class instance with object detection and classifier models using tensorflow toolset
        :param homedir: directory models, and labels
        :param object_model: file name of the object detection model
        :param object_model_labels: file name of the labels the detection model uses
        :param classifier_model: file name of the classifier model
        :param classifier_labels: file name of the classifier models labels
        :param classifier_thresholds: thresholds to apply to each class for the probability of that class
        :param detect_object_min_confidence: min probability to consider for positive object detection
        :param screenheight: height of images
        :param screenwidth: width of images
        :param contrast_chg: desired shift in contrast, typical values are .8 to 1.2 with 1.o being neutral
        :param color_chg: desired shift in color
        :param brightness_chg: desired shift in brightness
        :param sharpness_chg: desired shift in sharpness
        :param min_img_percent: min percent of the picture that the obj should take up for classification
        :param target_object: what object are we looking for: bird, plane, superman?
        :param classify_object_min_confidence: min confidence to consider for any classification
        :param debug: flag for debugging, provides extra print
        :param output_class: object class used to handle output for logging, printing, etc.
        """
        self.detector_file = homedir + object_model  # object model
        self.detector_labels_file = homedir + object_model_labels  # obj model label
        self.target_objects = static_functions.convert_to_list(target_object)
        self.target_object_found = False
        self.classifier_file = homedir + classifier_model  # classifier model
        self.labels = classifier_labels
        self.thresholds = classifier_thresholds
        self.classifier_labels_file = homedir + classifier_labels
        self.classifier_thresholds_file = homedir + classifier_thresholds
        # load the last col in the file only as a set of int values.  900 = .900
        # genfromtxt may behave differently on the pi than windows
        # self.classifier_thresholds = np.genfromtxt(self.classifier_thresholds_file, delimiter=',', usecols=[-1])
        self.classifier_thresholds = []
        with open(self.classifier_thresholds_file, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                self.classifier_thresholds.append(row[1])

        self.detector, self.obj_detector_possible_labels, self.detector_is_floating_model = (
            self.init_tf2(self.detector_file, self.detector_labels_file))
        self.classifier, self.classifier_possible_labels, self.classifier_is_floating_model = (
            self.init_tf2(self.classifier_file, self.classifier_labels_file))
        self.detect_obj_min_confidence = detect_object_min_confidence
        self.classify_object_min_confidence = classify_object_min_confidence
        self.obj_confidence = 0
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
        self.color_index = 0
        self.text_color_index = 0
        self.set_colors()  # sets color index and text_color index

        # set image adjustment parameters
        self.contrast_chg = contrast_chg
        self.brightness_chg = brightness_chg
        self.color_chg = color_chg
        self.sharpness_chg = sharpness_chg
        self.min_img_percent = min_img_percent
        self.screenwidth = screenwidth
        self.screenheight = screenheight
        self.screen_sq_pixels = screenwidth * screenheight
        # self.img = Image.fromarray(np.zeros((screenheight, screenwidth, 3), dtype=np.uint8))  # null image
        self.img = Image.new('RGB', (screenwidth, screenheight), color='black')  # null image at startup
        self.debug = debug
        self.output_class = output_class
        self.output_function = output_class.message if output_class is not None else None
        return

    def init_tf2(self, model_file: str, label_file_name: str) -> tuple:
        """
        initialize a tensorflow object for inference, attempt lite version first for Rasp PI
        if not present look for the full version
        :param model_file: name of the model to load for this instance of tensorflow
        :param label_file_name: name of the corresponding labels for the model
        :return: tuple of the tensorflow interpreter, list of the labels, and a boolean if
            the model is float32 (True) or int (False)
        """
        possible_labels = np.asarray(self.load_labels(label_file_name))  # load label file and convert to list
        try:  # load tensorflow lite on rasp pi
            interpreter = tflite.Interpreter(model_file, None)
        except NameError:  # load full tensor for desktop dev
            interpreter = tf.lite.Interpreter(model_file, None)
        interpreter.allocate_tensors()
        is_floating_model = interpreter.get_input_details()[0]['dtype'] == np.float32  # is float32
        print(f'Requested model is {model_file} and this model is a float model: {is_floating_model}')
        return interpreter, possible_labels, is_floating_model

    def output_ctl(self, message: str, msg_type: str = '') -> None:
        """
        functions sole purpose is to allow testing with the print function instead of the web send in the full app
        :param message: message to print, log and/or send to web
        :param msg_type: message type for web processing and log
        :return: None
        """
        if self.output_function is not None:
            self.output_function(message=message, msg_type=msg_type)
        else:
            print(message)
        return

    @staticmethod
    def load_labels(filename: str) -> list:
        """
        load label file for obj detection or classification model
        :param filename: name of file containing labels
        :return: list of labels
        """
        with open(filename, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def detect(self, detect_img: Image.Image) -> bool:
        """
        function performs object detection for the target object. Converts the image to a TF format for inference
        :param detect_img: Pillow Img from camera
        :return: true if target object was found in the image, useful for processing in a loop for detected objects
        """
        self.img = detect_img.copy()
        self.detected_confidences = []
        self.detected_labels = []  # possible object labels
        self.detected_rects = []
        self.target_object_found = False

        input_details = self.detector.get_input_details()
        output_details = self.detector.get_output_details()
        input_data = self.convert_img_to_tf(self.img, input_details, self.detector_is_floating_model)
        self.detector.set_tensor(input_details[0]['index'], input_data)
        self.detector.invoke()
        if self.detector_is_floating_model is False:  # tensor lite obj detection prebuilt model
            det_rects = self.detector.get_tensor(output_details[0]['index'])
            det_labels_index = self.detector.get_tensor(output_details[1]['index'])  # label array for each result
            det_confidences = self.detector.get_tensor(output_details[2]['index'])
            # if self.debug:
            #     print(f'det_class_label.py model results {det_labels_index}, {det_confidences}')
            for index, self.obj_confidence in enumerate(det_confidences[0]):
                labelidx = int(det_labels_index[0][index])  # get result label index for labels;
                det_label = self.obj_detector_possible_labels[labelidx]  # grab text from possible labels
                if self.debug and 'bird' in det_label:
                    print(f'det_class_label.py detect: for index {index} label is {det_label} '
                          f'with confidence {self.obj_confidence} and threshold is {self.detect_obj_min_confidence}. '
                          f'Target object in {det_label in self.target_objects}')
                if self.obj_confidence >= self.detect_obj_min_confidence and \
                        det_label in self.target_objects:
                    self.target_object_found = True
                    self.detected_confidences.append(self.obj_confidence)
                    self.detected_labels.append(det_label)
                    self.detected_rects.append(det_rects[0][index])
        else:
            self.output_function('det_class_label.py detect received a floating model and is only programmed for int')
        return self.target_object_found

    def classify(self, class_img: Image.Image, use_confidence_threshold: bool = True) -> float:
        """
        loop over detected objects and classify each one.  Note there may be more than one bird in the image
        make an equalized img and compare results from both
        uses self.detected_confidences,detected_labels, detected_rects lists
        stash prior classification for later processing if current frame does not find a bird
        :param class_img: img to perform classification on
        :param use_confidence_threshold: True requires any classification confidence to exceed threshold
        :return: The highest confidence / probability for a class across all proposed classes
        """
        self.classified_rects = []
        self.classified_rects_area = []
        self.classified_confidences = []
        self.classified_labels = []
        max_confidence = 0  # set to zero in case we do n0t get a classification match
        for i, det_confidence in enumerate(self.detected_confidences):  # loop over detected target objects
            (start_x, start_y, end_x, end_y) = self.scale_rect(class_img, self.detected_rects[i])  # set x,y bound box
            rect = (start_x, start_y, end_x, end_y)
            rect_percent_scr = ((end_x - start_x) * (end_y - start_y)) / self.screen_sq_pixels * 100  # % of screen img
            if self.classifier_is_floating_model is True:  # normalize here current model is int so this will not exec
                class_img = image_proc.normalize(class_img)  # normalize img

            crop_img = class_img.crop((start_x, start_y, end_x, end_y))  # extract image for better classification
            classify_conf, classify_label = self.classify_obj(crop_img, use_confidence_threshold, rect_percent_scr)

            adjustedimg = image_proc.enhance(class_img, brightness=self.brightness_chg, contrast=self.contrast_chg,
                                             color=self.color_chg, sharpness=self.sharpness_chg)
            crop_adjustedimg = adjustedimg.crop((start_x, start_y, end_x, end_y))
            classify_conf_adjusted, classify_label_adjusted = (
                self.classify_obj(crop_adjustedimg, use_confidence_threshold, rect_percent_scr))

            # take the best result between cropped img and adjusted cropped img
            classify_label = classify_label if classify_conf >= classify_conf_adjusted else classify_label_adjusted
            classify_conf = classify_conf if classify_conf >= classify_conf_adjusted else classify_conf_adjusted
            # if there was a detection print a message and append the label to findings
            if len(classify_label.strip()) > 0 and classify_conf != 0:
                self.output_ctl(message=f'match returned: confidence {classify_conf:.3f}, {classify_label},',
                                msg_type='match')
                self.classified_labels.append(classify_label)
                self.classified_confidences.append(classify_conf)
                self.classified_rects.append(rect)
                self.classified_rects_area.append(rect_percent_scr)

        # done processing detect objects, check for a classification match (non-zero finding)
        # set last known to current species if confidence is above threshold
        if max(self.classified_confidences, default=0) != 0:  # not empty list or zero
            max_confidence = max(self.classified_confidences)
            self.last_known_classified_rects = self.classified_rects
            self.last_known_classified_rects_area = self.classified_rects_area
            self.last_known_classified_labels = self.classified_labels
            self.last_known_classified_confidences = self.classified_confidences
        return max_confidence

    def classify_obj(self, class_img: Image.Image, use_confidence_threshold: bool = True,
                     rect_percent_scr: float = 100.00) -> tuple:
        """
        take input image and return best result and label
        the function will sort the results and compare the confidence to the confidence for that label (species)
        i the ML models confidence is higher than the threshold for that label (species) it will stop searching and
        return that best result
        :param class_img: image containing object to classify
        :param use_confidence_threshold: requires probability for species returned to exceed threshold for valid result
        :param rect_percent_scr: percentage of screen bounding box is of the total image
        :return: a tuple that contains the best confidence result and best label for requested classification
        """
        maxcresult = float(0)  # max confidence aka prediction result from model
        maxlresult = ''  # best label from max confidence
        # grab the input details setup at init.  use that clean version for further processing
        input_details = self.classifier.get_input_details()
        input_data = self.convert_img_to_tf(class_img, input_details, self.classifier_is_floating_model)
        self.classifier.set_tensor(input_details[0]['index'], input_data)  # setup batch of 1 image
        self.classifier.invoke()  # inference
        output_details = self.classifier.get_output_details()[0]  # get results values as floats .9 = 90%
        output = np.squeeze(self.classifier.get_tensor(output_details['index']))  # remove all 1 dim to get this to list
        # if self.debug:
        #     print(f'det_class_label.py classify obj: output was {output}')
        # If the model is quantized aka tflite uint8 data (not a floating pt model) then de-quantize the results
        if self.classifier_is_floating_model is False:
            scale, zero_point = output_details['quantization']
            output = scale * (output - zero_point)  # scale factor to adjust results, this had a *10 is that need on pi?
            # if self.debug:
            #     print(f'det_class_label.py classify obj: adjusted output was {output} using scale {scale} '
            #           f'and zero point {zero_point}')
        cindex = np.argpartition(output, -10)[-10:]  # output is an array with many zeros find index for nonzero values
        # loop over top N results to find best match; highest score align with matching species threshold
        for lindex in cindex:
            lresult = str(self.classifier_possible_labels[lindex]).strip()  # grab label,push to string instead of tuple
            cresult = float(output[lindex]) if float(output[lindex]) > 0 else 0
            cresult = cresult - math.floor(cresult) if cresult > 1 else cresult  # ignore whole numbers, keep decimals
            if self.check_threshold(cresult, lindex, rect_percent_scr, use_confidence_threshold):
                if cresult > maxcresult:  # if this above threshold and is a better confidence result store it
                    maxcresult = cresult
                    maxlresult = lresult
        if self.debug:
            print(f'det_class_label.py classify obj: final answer was {maxcresult} and {maxlresult}')
        return maxcresult, maxlresult  # highest confidence with best match

    @staticmethod
    def convert_img_to_tf(pil_img: Image.Image, input_details, is_floating_model: bool):
        """
        takes a PIL image type and converts it to np array for tensor and resizes for classification model
        TF models can be full or lite versions.  The full version uses floating point numbers.  Lite version
        is integer only so the data must be converted accordingly
        :param pil_img: image to convert for tensor flow model
        :param input_details: tf. get_input_details object used for dtype and shape
        :param is_floating_model: true if the model is float32 false if the model is int
        :return: input details for model
        """
        height = input_details[0]['shape'][1]  # NxHxWxC, H:1, W:2
        width = input_details[0]['shape'][2]
        reshape_image = pil_img.resize((width, height))  # shape to match model input requirements
        image_np = np.array(reshape_image)  # convert to np array
        # TF models expect input as a batch even if a single image.
        image_np_expanded = np.expand_dims(image_np, axis=0)  # adds an extra dimension creating a batch size of 1

        # model could be full or lite version.  lite version in int only.  full version is float32
        # integer normalization is not attempted since much of the data would be lost after the decimal
        input_data = image_np_expanded.astype('uint8') if is_floating_model is False \
            else image_np_expanded.astype('float32')
        return input_data

    @staticmethod
    def scale_rect(scale_rect_img: Image.Image, box: tuple) -> tuple:
        """
        scale the bounding box to the image and ensure that the values are in the np array index
        :param scale_rect_img: Pillow Image
        :param box: tuple containing starting x and y and ending x and y coordinates for bounding box of object
        :return: tuple of new coordinates
        """
        img_width, img_height = scale_rect_img.size
        y_min = int(max(1, (box[0] * img_height)))
        x_min = int(max(1, (box[1] * img_width)))
        y_max = int(min(img_height, (box[2] * img_height)))
        x_max = int(min(img_width, (box[3] * img_width)))
        return x_min, y_min, x_max, y_max

    def add_boxes_and_labels(self, label_img: Image.Image, use_last_known: bool = False) -> Image.Image:
        """
        add bounding box and label to an image.  May have a rect with no species and a zero confidence,
        in that case uses the last known label and confidence
        :param label_img: image to draw box and write label on
        :param use_last_known: uses the last known species, used in animated gif maker is a label is unknown for a frame
        :return: labeled image + bound box and text
        """
        # set local variables for processing
        if use_last_known and round(max(self.classified_confidences, default=0), 2) == 0:
            classified_rects = self.last_known_classified_rects
            classified_rects_area = self.last_known_classified_rects_area
            classified_labels = self.last_known_classified_labels
            classified_confidences = self.last_known_classified_confidences
        else:
            classified_rects = self.classified_rects
            classified_rects_area = self.classified_rects_area
            classified_labels = self.classified_labels
            classified_confidences = self.classified_confidences

        # draw on the image
        draw = PILImageDraw.Draw(label_img)
        font = draw.getfont()
        c_labels_list_len = len(classified_labels)
        for i, rect in enumerate(classified_rects):
            (start_x, start_y, end_x, end_y) = rect
            # the classifier may return more rectangles than labels.  in that case apply the last label
            # and the last confidence.  Wrap this in a try except to trap any unhandled errors
            try:
                classified_label = classified_labels[i] if c_labels_list_len > i - 1 else classified_labels[-1]
                classified_confidence = classified_confidences[i] if c_labels_list_len > i - 1 \
                    else classified_confidences[-1]

                # add text to top and bottom of image, make box slightly large and put text on top and bottom
                draw_text_label = f'{static_functions.common_name(classified_label)} {classified_confidence * 100:.2f}%'
                draw.text((start_x, start_y-50), draw_text_label, font=font, fill='white')
                draw.text((start_x, end_y+50), draw_text_label, font=font, fill='white')
                # line fill is a tuple of RGB values
                draw.line([(start_x-25, start_y-25), (start_x-25, end_y+25), (start_x-25, end_y+25),
                           (end_x+25, end_y+25), (end_x+25, end_y+25),
                           (end_x+25, start_y-25), (end_x+25, start_y-25), (start_x-25, start_y-25)],
                          fill=tuple(self.colors[self.get_next_color(from_index=i)]), width=2)
            except Exception as e:
                print('tried drawing text on the image of ith rectangle with rect:', i)
                print(e)
                print(classified_labels, classified_confidences, classified_rects, classified_rects_area)
        return label_img

    def set_colors(self) -> None:
        """
        sets color values for bounding boxes and text based on random picks
        :return: none
        """
        self.color_index = random.randint(0, (len(self.colors)) - 1)
        self.text_color_index = random.randint(0, (len(self.colors)) - 1)
        return

    def get_next_color(self, from_index: int = 0) -> int:
        """
        Overlapping objects need their own colors for bounding boxes.  This functions finds the next color
        :param from_index: current color in use, don't reuse this, grab the next one.
        :return: int containing the next index for a color, use tuple(self.color[return]) as the fill for draw line
        """
        return (self.color_index + from_index) % (len(self.colors) - 1)

    def check_threshold(self, cresult: float, lindex: int, rect_percent_scr: float ,
                        use_confidence_threshold: bool=False) -> bool:
        """
        checks the predictions confidence against the confidence thresholds to see if the species
        is allowed in this geography or has a custom setting to allow for more or less positives.
        classifier_thresholds are a percentage * 10 (no decimals) so values range from 0 to 1000.
        we must multiply other values by 1000 to get same scale.
        rules
        1. threshold cannot be -1 (not present in geolocation), returns false
        2. img must take up the specified min % of the image, or it is tossed out, returns false
        3. use_confidence_threshold false results in the function always returning true.  useful for debugging
        4. prediction must be equal or exceed minimum score
        :param cresult: predicted confidence, float  between 0 and 1;
        :param lindex: integer containing the index of the species predicted for look up in threshold list
        :param use_confidence_threshold: requires classification prob to exceed threshold for validity
        :param rect_percent_scr: min screen percentage object takes up in image.
        :return: true if prediction confidence is over threshold for species
        """
        label_threshold = 0
        # apply rules 1 and 2
        if self.debug:
            print(f'det_class_label.py check_threshold: confidence {cresult} for label index {lindex}, '
                  f'species threshold is {(self.classifier_thresholds[int(lindex)])} / 1000 with percent of img at'
                  f'{rect_percent_scr} and a threshold min image percent of {self.min_img_percent}'
                  f'and use threshold is {use_confidence_threshold}')
        if int(self.classifier_thresholds[int(lindex)]) == -1 or rect_percent_scr < self.min_img_percent:
            return False
        # apply rule 3
        elif use_confidence_threshold is False:  # the requester does not care to check the threshold
            return True
        # use default threshold if threshold is 0 else use species specific score
        try:  # handle typos in threshold file
            label_threshold = float(self.classify_object_min_confidence * 1000
                                    if self.classifier_thresholds[int(lindex)] == 0
                                    else self.classifier_thresholds[int(lindex)])
        except Exception as e:
            print(e)
            print(f'det_class_label.py check_threshold error, possible type in input file for:'
                  f'{self.classifier_thresholds[int(lindex)]}')  # where was the error in the file?
            print(cresult)  # what was the prediction
            cresult = 0  # cause a false to be returned for this species on an error
        return cresult > 0 and cresult >= (float(label_threshold) / 1000) and int(label_threshold) != -1

    def get_obj_data(self) -> tuple:
        """
        return last objects information
        method returns rectangles, labels, and confidences for the last detected object
        :return: tuple of lists
        """
        return self.classified_rects, self.classified_labels, self.classified_confidences

    def set_ojb_data(self, classified_rects: list, classified_labels: list, classified_confidences: list) -> None:
        """
        reset info to earlier bird data, used by bird class to revert to the first bird in a sequence
        :param classified_rects: list of tuples containing 4 int representing the rect containing obj in image
        :param classified_labels: list of labels matching rectangles
        :param classified_confidences: list of confidences matching rects
        :return:
        """
        self.classified_rects = classified_rects
        self.classified_labels = classified_labels
        self.classified_confidences = classified_confidences
        return


if __name__ == '__main__':
    label = ''
    debugb = True
    img_test = Image.open('/home/pi/birdclass/1.jpg')
    birds = DetectClassify(homedir='c:/Users/jimma/PycharmProjects/birdclassifier/', debug=debugb)
    birds.detect(img_test)  # run object detection

    print('main testing code: objects detected', birds.detected_confidences)
    print('main testing code: labels detected', birds.detected_labels)
    print('main testing code: rectangles', birds.detected_rects)

    birds.classify(birds.img)  # classify species
    print(f'main testing code: {birds.classified_labels}')
    if len(birds.classified_labels) == 0:
        label = "main testing code: no classification text"
    else:
        label = birds.classified_labels[0]
    print(f'main testing code: final label is {label}')
    img = birds.add_boxes_and_labels(img_test)
    img.save('imgtest.jpg')
    img.show()
