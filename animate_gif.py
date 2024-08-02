# MIT License
#
# 2024 Jim Maastricht
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
# built by JimMaastricht5@gmail.com
import image_proc
from datetime import datetime
from collections import defaultdict


# default dictionary returns a tuple of zero confidence and zero bird count
def default_value():
    return 0


class BirdGif:
    def __init__(self, motion_detector_cls, birds_cls, gcs_storage_cls, brightness_chg=1.0, min_animated_frames=10):
        self.motion_detect = motion_detector_cls
        self.birds = birds_cls
        self.gcs_storage = gcs_storage_cls
        self.brightness_chg = brightness_chg
        self.min_animated_frames = min_animated_frames

        # init dictionaries
        self.census_dict = defaultdict(default_value)  # track all results and pick best confidence
        self.confidence_dict = defaultdict(default_value)  # track all results and pick best confidence
        self.weighted_dict = defaultdict(default_value)  # conf * (count -1) to put emphasis on # of observations

        # init attributes of a gift
        self.local_gif_filename = ''
        self.gcs_gif_filename = ''
        self.animated = False
        self.best_label = ''
        self.best_confidence = 0
        self.frames_with_birds = 0

        return

    def convert_to_list(self, input_str_list):
        return input_str_list if isinstance(input_str_list, list) else [input_str_list]

    def common_name(self, name):
        cname, sname = '', ''
        try:
            sname = str(name)
            sname = sname[sname.find(' ') + 1:] if sname.find(' ') >= 0 else sname  # remove index number
            sname = sname[0: sname.find('[') - 1] if sname.find('[') >= 0 else sname  # remove sex
            cname = sname[sname.find('(') + 1: sname.find(')')] if sname.find('(') >= 0 else sname  # common name
        except Exception as e:
            print(e)
        return cname

    # should be passing default dictionary
    # keeps track of count in label_dict (census) and confidences totals in conf_dict
    # weighted dict is confidence total * (count-1) to weight away from low # of occurrences
    def build_dict(self):
        labels_list = self.convert_to_list(self.birds.classified_labels)
        confidences_list = self.convert_to_list(self.birds.classified_confidences,)
        for ii, label in enumerate(labels_list):
            self.census_dict[label] += 1
            self.confidence_dict[label] += confidences_list[ii]
            self.weighted_dict[label] = (self.census_dict[label] - 1) * self.confidence_dict[label]
        return

    def build_gif(self, event_count, first_img_jpg):
        # grab a stream of pictures, add first pic from above, and build animated gif
        # return gif, filename, animated boolean, and best label as the max of all confidences
        self.local_gif_filename, self.gcs_gif_filename, self.best_label, self.best_confidence = ('', '', '', 0)
        labeled_frames = []
        self.animated = False  # set to true if min # of frames captured with birds
        gif = first_img_jpg  # set a default if animated = False
        last_good_frame = 0  # find last frame that has a bird, index zero is good based on first image
        self.frames_with_birds = 1  # count of frames with birds, set to 1 for first img
        self.census_dict = defaultdict(default_value)  # track all results and pick best confidence
        self.confidence_dict = defaultdict(default_value)  # track all results and pick best confidence
        self.weighted_dict = defaultdict(default_value)  # conf * (count -1) to put emphasis on # of observations
        self.build_dict()  # use census, confidence,weighted, bird labels and bird confidence

        frames = self.motion_detect.capture_stream()  # capture a list of images
        first_img_jpg = self.birds.add_boxes_and_labels(img=first_img_jpg)
        labeled_frames.insert(0, image_proc.convert_image(img=first_img_jpg))  # isrt 1st img
        for i, frame in enumerate(frames):
            frame = image_proc.enhance(img=frame, brightness=self.brightness_chg)
            if self.birds.detect(img=frame):  # find bird object in frame and set rectangles containing object
                if self.birds.classify(img=frame, use_confidence_threshold=False) > 0:   # classify at rect & chk conf
                    self.frames_with_birds += 1
                    last_good_frame = i + 1  # found a bird, add one to frame to account for insert of 1st image
                self.build_dict()  # use census, confidence,weighted, bird labels and bird confidence
            labeled_frames.append(self.birds.add_boxes_and_labels(img=frame, use_last_known=True))

        # build confidence, label, and weighted label
        self.best_confidence = self.confidence_dict[max(self.confidence_dict, key=self.confidence_dict.get)] / \
            self.census_dict[max(self.confidence_dict, key=self.confidence_dict.get)]  # sum conf/bird cnt
        self.best_label = max(self.confidence_dict, key=self.confidence_dict.get)
        best_weighted_label = max(self.weighted_dict, key=self.weighted_dict.get)

        # if bird is in min number of frames build gif
        if self.frames_with_birds >= (self.min_animated_frames - 1):
            gif, self.local_gif_filename = image_proc.save_gif(frames=labeled_frames[0:last_good_frame])
            self.gcs_gif_filename = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}{str(event_count)}' \
                f'({self.common_name(best_weighted_label).replace(" ", "")}).gif'  # space from img name
            self.gcs_storage.send_file(name=self.gcs_gif_filename, file_loc_name=self.local_gif_filename)
            self.animated = True

        return gif


# old code....
# def best_confidence_and_label(census_dict, confidence_dict):
#     best_confidence, best_confidence_label, best_census, best_census_label = 0, '', 0, ''
#     try:
#         census_dict, confidence_dict = remove_single_observations(census_dict, confidence_dict)  # multishots results
#         best_confidence = confidence_dict[max(confidence_dict, key=confidence_dict.get)] / \
#             census_dict[max(confidence_dict, key=confidence_dict.get)]  # sum conf/bird cnt
#         best_confidence_label = max(confidence_dict, key=confidence_dict.get)
#         best_census = census_dict[max(census_dict, key=census_dict.get)]
#         best_census_label = max(census_dict, key=census_dict.get)
#     except Exception as e:
#         print(e)
#     # print('best confidence:', best_confidence, best_confidence_label)
#     # print('best census:', best_census, best_census_label)
#     # if best_confidence_label != best_census_label:
#     #  what to do here?  Which one is better?  High count or high confidence?
#     return best_confidence, best_confidence_label, best_census, best_census_label
