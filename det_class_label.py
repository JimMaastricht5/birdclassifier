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
import numpy as np
import random
from PIL import ImageDraw as PILImageDraw
from PIL import Image
import image_proc
import math
import static_functions


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
                 classifier_thresholds: str = 'coral.ai.inat_bird_threshold.csv',
                 detect_object_min_confidence: float = .9, screenheight: int = 480, screenwidth: int = 640,
                 contrast_chg: float = 1.0, color_chg: float = 1.0,
                 brightness_chg: float = 1.0, sharpness_chg: float = 1.0,
                 min_img_percent: float = 10.0, target_object: str = 'bird',
                 classify_object_min_confidence: float = .8, output_class=None) -> None:
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
        :param output_class: object class used to handle output for logging, printing, etc.
        """
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
        self.output_class = output_class
        self.output_function = output_class.message
        return

    def init_tf2(self, model_file: str, label_file_name: str) -> tuple:
        """
        initialize a tensorflow object for inference, attempt lite version first for Rasp PI
        if not present look for the full version
        :param model_file: name of the model to load for this instance of tensorflow
        :param label_file_name: name of the corresponding labels for the model
        :return: tuple of the tensorflow interpreter and a list of the labels
        """
        possible_labels = np.asarray(self.load_labels(label_file_name))  # load label file and convert to list
        try:  # load tensorflow lite on rasp pi
            interpreter = tflite.Interpreter(model_file, None)
        except NameError:  # load full tensor for desktop dev
            interpreter = tf.lite.Interpreter(model_file, None)
        interpreter.allocate_tensors()
        return interpreter, possible_labels

    def output_class(self, message: str, msg_type: str = ''):
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
        :return: true if target object was found in the image
        """
        self.img = detect_img.copy()
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
                det_label = self.obj_detector_possible_labels[labelidx]  # grab text from possible labels
                if self.obj_confidence >= self.detect_obj_min_confidence and \
                        det_label in self.target_objects:
                    self.target_object_found = True
                    self.detected_confidences.append(self.obj_confidence)
                    self.detected_labels.append(det_label)
                    self.detected_rects.append(det_rects[0][index])
        return self.target_object_found

    def classify(self, class_img: Image.Image, use_confidence_threshold: bool = True) -> float:
        """
    # make classifications using img and detect results
    # compare class_img and equalized images
    # use self.detected_confidences,detected_labels, detected_rects lists
    # stash prior classification for later processing if current frame does not find a bird
        :param class_img:
        :param use_confidence_threshold: True requires any classification confidence to exceed threshold
        :return: The highest confidence / probability for a class across all proposed classes
        """
        self.classified_rects = []
        self.classified_rects_area = []
        self.classified_confidences = []
        self.classified_labels = []
        prior_rect = (0, 0, 0, 0)
        for i, det_confidence in enumerate(self.detected_confidences):  # loop over detected target objects
            (startX, startY, endX, endY) = self.scale_rect(class_img, self.detected_rects[i])  # set x,y bounding box
            rect = (startX, startY, endX, endY)
            rect_percent_scr = ((endX - startX) * (endY - startY)) / self.screen_sq_pixels * 100  # % of screen of img
            # ?? note this section needs review for improvement crop, pad, equalize, normalize....
            crop_img = class_img.crop((startX, startY, endX, endY))  # extract image for better classification
            equalizedimg = image_proc.enhance(class_img, brightness=self.brightness_chg, contrast=self.contrast_chg,
                                              color=self.color_chg, sharpness=self.sharpness_chg)
            crop_equalizedimg = equalizedimg.crop((startX, startY, endX, endY))
            classify_conf, classify_label = self.classify_obj(crop_img, use_confidence_threshold,
                                                              rect_percent_scr)
            classify_conf_equalized, classify_label_equalized = self.classify_obj(crop_equalizedimg,
                                                                                  use_confidence_threshold,
                                                                                  rect_percent_scr)
            # take the best result between cropped img and enhanced cropped img
            classify_label = classify_label if classify_conf >= classify_conf_equalized else classify_label_equalized
            classify_conf = classify_conf if classify_conf >= classify_conf_equalized else classify_conf_equalized
            if classify_conf != 0:
                self.output_class(message=f'match returned: confidence {classify_conf:.3f}, {classify_label},',
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

    def classify_obj(self, class_img, use_confidence_threshold=True, screen_percent=100.00):
        """
    # input image and return best result and label
    # the function will sort the results and compare the confidence to the confidence for that label (species)
    # if the ML models confidence is higher than the threshold for that label (species) it will stop searching and
    # return that best result
        :param class_img:
        :param use_confidence_threshold: requires probability for species returned to exceed threshold for valid result
        :param screen_percent:
        :return:
        """
        input_details = self.classifier.get_input_details()
        floating_model, input_data = self.convert_img_to_tf(class_img, input_details)
        self.classifier.set_tensor(input_details[0]['index'], input_data)
        self.classifier.invoke()  # invoke classification
        output_details = self.classifier.get_output_details()[0]
        output = np.squeeze(self.classifier.get_tensor(output_details['index']))
        # If the model is quantized (tflite uint8 data), then dequantize the results
        if output_details['dtype'] == np.uint8:
            scale, zero_point = output_details['quantization']
            output = scale * (output - zero_point) * 10  # scale factor to adjust results
        cindex = np.argpartition(output, -10)[-10:]
        # loop over top N results to find best match; highest score align with matching species threshold
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
        return maxcresult, maxlresult  # highest confidence with best match

    def convert_img_to_tf(self, pil_img, input_details):
        """
        takes a PIL image type and converts it to np array for tensor and resizes for classification model
        :param pil_img: image to convert for tensor flow model
        :param input_details: tf. get_input_details object used for dtype and shape
        :return:
        """
        # check the type of the input tensor
        floating_model = input_details[0]['dtype'] == np.float32  # is float32
        height = input_details[0]['shape'][1]  # NxHxWxC, H:1, W:2
        width = input_details[0]['shape'][2]
        reshape_image = pil_img.resize((width, height))  # shape to match model input requirements
        image_np = np.array(reshape_image)  # convert to np array
        # TF models expect input as a batch even if a single image.  Pad for N
        image_np_expanded = np.expand_dims(image_np, axis=0)

        if floating_model:
            input_data = image_np_expanded.astype('float32')
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std
        else:
            input_data = image_np_expanded.astype('uint8')
        return floating_model, input_data

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

    def add_boxes_and_labels(self, label_img, use_last_known=False):
        """
    # add bounding box and label to an image
    # we may have a rect with no species and a zero confidence, in that case use the last known label and confidence
        :param label_img:
        :param use_last_known:
        :return:
        """
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

        draw = PILImageDraw.Draw(label_img)
        font = draw.getfont()
        c_labs_len = len(classified_labels)
        for i, rect in enumerate(classified_rects):
            (start_x, start_y, end_x, end_y) = rect
            try:  # add text to top and bottom of image, make box slightly large and put text on top and bottom
                # sometimes the label and conf are the same for more than one rect, handle that here....
                classified_label = classified_labels[i] if c_labs_len > i - 1 else classified_labels[-1]
                classified_confidence = classified_confidences[i] if c_labs_len > i - 1 else classified_confidences[-1]
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

    def check_threshold(self, cresult, lindex, use_confidence_threshold, screen_percent):
        """
    # func checks threshold by each label passed as a nparray with text in col 0 and threshold in col 1
    # object cannot be -1 (not present in geo location), confidence cannot be 0,must be equal or exceed minimum score
    # img must take up the specified min % of the image, or it is tossed out
    # cresult is a decimal % 0 - 1; lindex is % * 10 (no decimals) must div by 1000 to get same scale
        :param cresult:
        :param lindex:
        :param use_confidence_threshold: requires classification prob to exceed threshold for validity
        :param screen_percent:
        :return:
        """
        # grab default if species has 0 confidence, else use species specific score
        label_threshold = self.classify_object_min_confidence * 1000 \
            if self.classifier_thresholds[int(lindex)][1] == 0 \
            else self.classifier_thresholds[int(lindex)][1]
        # push to zero if the use_threshold boolean is false, this automatically puts any confidence over the threshold
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
        return (float(label_threshold) != -1 and screen_percent >= self.min_img_percent and
                cresult > 0 and cresult >= float(label_threshold) / 1000)

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
    img_test = Image.open('/home/pi/birdclass/0.jpg')
    birds = DetectClassify('c:/Users/jimma/PycharmProjects/birdclassifier/', detect_object_min_confidence=.9)
    birds.detect(img_test)  # run object detection

    print('objects detected', birds.detected_confidences)
    print('labels detected', birds.detected_labels)
    print('rectangles', birds.detected_rects)

    birds.classify(birds.img)  # classify species
    print(birds.classified_labels)
    label = birds.classified_labels[0]
    if len(label) == 0:
        label = "no classification text"
    print(label)
    img = birds.add_boxes_and_labels(img_test)
    img.save('imgtest.jpg')
    img.show()

# old code
# def pick_a_color(self) -> int:
#     """
#     picks a random color for bounding boxes
#     :return: integer for the color
#     """
#     return random.randint(0, (len(self.colors)) - 1)
    # set label for box in image use short species name instead of scientific name
    # def label_text(self, label, confidence):
    #     """
    #
    #     :param label:
    #     :param confidence:
    #     :param screen_percent:
    #     :return:
    #     """
    #     sname = str(label)  # make sure label is considered a string
    #     start = sname.find('(') + 1  # find start of common name, move one character to drop (
    #     end = sname.find(')')
    #     cname = sname[start:end] if start >= 0 and end >= 0 else sname
    #     common_name = f'{cname} {confidence * 100:.2f}%'
    #     return common_name
