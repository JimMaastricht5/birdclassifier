import image_proc
from datetime import datetime
from collections import defaultdict


def default_value():
    return 0


class Animate:
    def __init__(self):
        return

    # default dictionary returns a tuple of zero confidence and zero bird count

    def convert_to_list(self, input_str_list):
        return input_str_list if isinstance(input_str_list, list) else [input_str_list]

    # should be passing default dictionary
    # keeps track of count in label_dict (census) and confidences totals in conf_dict
    # weighted dict is confidence total * (count-1) to weight away from low # of occurrences
    def build_dict(self, label_dict, input_labels_list, conf_dict, input_confidences_list, weighted_dict):
        labels_list = self.convert_to_list(input_labels_list)
        confidences_list = self.convert_to_list(input_confidences_list)
        for ii, label in enumerate(labels_list):
            label_dict[label] += 1
            conf_dict[label] += confidences_list[ii]
            weighted_dict[label] = (label_dict[label] - 1) * conf_dict[label]
        return label_dict, conf_dict, weighted_dict

    # loop through keys and remove census entries with 1 or zero observations
    def remove_single_observations(self, census_dict, conf_dict):
        key_list = []
        [key_list.append(key) if census_dict[key] <= 1 else '' for key in census_dict]
        [census_dict.pop(key) for key in key_list]
        [conf_dict.pop(key) for key in key_list]
        return census_dict, conf_dict

    def common_name(self, name):
        cname, sname = '', ''
        try:
            sname = str(name)
            sname = sname[sname.find(' ') + 1:] if sname.find(' ') >= 0 else sname  # remove index number
            # sex = sname[sname.find('[') + 1: sname.find(']')] if sname.find('[') >= 0 else ''  # retrieve sex
            sname = sname[0: sname.find('[') - 1] if sname.find('[') >= 0 else sname  # remove sex
            cname = sname[sname.find('(') + 1: sname.find(')')] if sname.find('(') >= 0 else sname  # common name
        except Exception as e:
            print(e)
        return cname

    def build_bird_animated_gif(self, args, motion_detect, birds, gcs_storage, event_count, first_img_jpg):
        # grab a stream of pictures, add first pic from above, and build animated gif
        # return gif, filename, animated boolean, and best label as the max of all confidences
        local_gif_filename, gcs_gif_filename, best_label, best_confidence, labeled_frames = '', '', '', 0, []
        animated = False  # set to true if min # of frames captured with birds
        gif = first_img_jpg  # set a default if animated = False
        last_good_frame = 0  # find last frame that has a bird, index zero is good based on first image
        frames_with_birds = 1  # count of frames with birds, set to 1 for first img
        census_dict = defaultdict(default_value)  # track all results and pick best confidence
        confidence_dict = defaultdict(default_value)  # track all results and pick best confidence
        weighted_dict = defaultdict(default_value)  # conf * (count -1) to put emphasis on # of observations
        census_dict, confidence_dict, weighted_dict = self.build_dict(census_dict, birds.classified_labels,
                                                                      confidence_dict, birds.classified_confidences,
                                                                      weighted_dict)
        frames = motion_detect.capture_stream()  # capture a list of images
        first_img_jpg = birds.add_boxes_and_labels(img=first_img_jpg)
        labeled_frames.insert(0, image_proc.convert_image(img=first_img_jpg, target='gif'))  # isrt 1st img
        for i, frame in enumerate(frames):
            frame = image_proc.enhance_brightness(img=frame, factor=args.brightness_chg)
            if birds.detect(img=frame):  # find bird object in frame and set rectangles containing object
                if birds.classify(img=frame, use_confidence_threshold=False) > 0:   # classify at rect & chk confidence
                    frames_with_birds += 1
                    last_good_frame = i + 1  # found a bird, add one to frame to account for insert of 1st image
                census_dict, confidence_dict, weighted_dict = self.build_dict(census_dict, birds.classified_labels,
                                                                              confidence_dict,
                                                                              birds.classified_confidences,
                                                                              weighted_dict)
            labeled_frames.append(birds.add_boxes_and_labels(img=frame, use_last_known=True))  # last label if unknown

        best_confidence = confidence_dict[max(confidence_dict, key=confidence_dict.get)] / \
            census_dict[max(confidence_dict, key=confidence_dict.get)]  # sum conf/bird cnt
        best_label = max(confidence_dict, key=confidence_dict.get)
        best_weighted_label = max(weighted_dict, key=weighted_dict.get)
        if frames_with_birds >= (args.minanimatedframes - 1):  # if bird is in min number of frames build gif
            gif, local_gif_filename = image_proc.save_gif(frames=labeled_frames[0:last_good_frame])
            gcs_gif_filename = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}{str(event_count)}' \
                               f'({self.common_name(best_weighted_label).replace(" ", "")}).gif'  # space from img name
            gcs_storage.send_file(name=gcs_gif_filename, file_loc_name=local_gif_filename)
            animated = True

        if best_label != best_weighted_label:
            print('--- Best label, confidence, and weight', best_label, best_confidence, best_weighted_label)
        return gif, local_gif_filename, gcs_gif_filename, animated, best_label, best_confidence, frames_with_birds
