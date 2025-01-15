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
#
# App to detect motion on a bird feeder, classify the object (is it a bird?),
# determine the species, record the appearance, send the image to twitter and the cloud for
# display. Web site at https://jimmaastricht5-tweetersp-1-main-veglvm.streamlit.app/
#
# uses tflite prebuilt google model for object detection and tensorflow lite model for bird classification
# Other models built with tensor flow  using tools from tensor in colab notebook
# https://colab.research.google.com/drive/1taZ9JincTaZuZh_JmBSC4pAbSQavxbq5#scrollTo=D3i_6WSXjUhk
# see read me for auth.py format requirements: https://github.com/JimMaastricht5/birdclassifier
# built by JimMaastricht5@gmail.com
import image_proc
import static_functions
from typing import Union
import det_class_label  # code to init tensor flow model and classify bird type, bird object
import motion_detector  # motion detector helper functions
import tweeter  # twitter helper functions
import population  # population census object, tracks species total seen and last time
import dailychores  # handle tasks that occur once per day or per hour
import weather
import output_stream
import argparse  # argument parser
import configparser  # read args from ini file
from datetime import datetime
import time
import os
import uuid
import gcs  # save objects to website for viewing
import animate_gif  # animate gif class


# default dictionary returns a tuple of zero confidence and zero bird count
def default_value():
    # default dictionary returns a zero if asked for a key that doesn't exist
    return 0


def process_args():
    """
    construct the argument parser and parse the arguments
    load settings from config file to allow for simple override
    parse arguments from arg parser list below
        "-cf", "--config_file", type=str, help='Config file'
        "-ol", "--offline", type=bool, default=False, help='Operate offline, do not transmit to cloud'
        "-db", "--debug", type=bool, default=False, help="debug flag"

        # camera settings
        "-fc", "--flipcamera", type=bool, default=False, help="flip camera image"
        "-sw", "--screenwidth", type=int, default=640, help="max screen width"
        "-sh", "--screenheight", type=int, default=480, help="max screen height"

        # general app settings
        "-gf", "--minanimatedframes", type=int, default=10, help="minimum number of frames with a bird"
        "-bb", "--broadcast", type=bool, default=False, help="stream images and text"
        "-v", "--verbose", type=bool, default=True, help="To tweet extra stuff or not"
        "-td", "--tweetdelay", type=int, default=1800
            help="Wait time between tweets is N species seen * delay/10 with not to exceed max of tweet delay"

        # motion and image processing settings,
        # note adjustments are used as both a detector second prediction and a final
        # adjustment to the output images.  # 1 no chg,< 1 -, > 1 +
        "-b", "--brightness_chg", type=int, default=1.2, help="brightness boost twilight"
        "-c", "--contrast_chg", type=float, default=1.0, help="contrast boost"  # 1 no chg,< 1 -, > 1 +
        "-cl", "--color_chg", type=float, default=1.0, help="color boost"  # 1 no chg,< 1 -, > 1 +
        "-sp", "--sharpness_chg", type=float, default=1.0, help="sharpness"  # 1 no chg,< 1 -, > 1 +

        # prediction defaults
        "-sc", "--species_confidence", type=float, default=.90, help="species confidence threshold"
        "-bc", "--bird_confidence", type=float, default=.6, help="bird confidence threshold"
        "-ma", "--minentropy", type=float, default=5.0, help="min change from first img to current to trigger motion"
        "-ms", "--minimgperc", type=float, default=10.0, help="ignore objects that are less than % of img"
        "-hd", "--homedir", type=str, default='/home/pi/birdclassifier/', help="home directory for files"
        "-la", "--labels", type=str, default='coral.ai.inat_bird_labels.txt', help="file for species labels "
        "-tr", "--thresholds", type=str, default='coral.ai.inat_bird_threshold.csv', help="file for species thresholds"
        "-cm", "--classifier", type=str, default='coral.ai.mobilenet_v2_1.0_224_inat_bird_quant.tflite',
            help="model name for species classifier"

        # feeder defaults
        "-ct", "--city", type=str, default='Madison,WI,USA', help="city name weather station uses OWM web service."
        '-fi', "--feeder_id", type=str, default=hex(uuid.getnode()), help='feeder id default MAC address'
        '-t', "--feeder_max_temp_c", type=int, default=86, help="Max operating temp for the feeder in C"
    :return: None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-cf", "--config_file", type=str, help='Config file')
    ap.add_argument("-ol", "--offline", type=bool, default=False,
                    help='Operate offline, do not transmit to cloud')
    ap.add_argument("-db", "--debug", type=bool, default=False, help="debug flag")

    # camera settings
    ap.add_argument("-fc", "--flipcamera", type=bool, default=False, help="flip camera image")
    ap.add_argument("-sw", "--screenwidth", type=int, default=480, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=640, help="max screen height")

    # general app settings
    ap.add_argument("-gf", "--minanimatedframes", type=int, default=10,
                    help="minimum number of frames with a bird")
    ap.add_argument("-bb", "--broadcast", type=bool, default=False, help="stream images and text")
    ap.add_argument("-v", "--verbose", type=bool, default=True, help="To tweet extra stuff or not")
    ap.add_argument("-td", "--tweetdelay", type=int, default=1800,
                    help="Wait time between tweets is N species seen * delay/10 with not to exceed max of tweet delay")

    # motion and image processing settings, note adjustments are used as both a detector second prediction and a final
    # adjustment to the output images.  # 1 no chg,< 1 -, > 1 +
    # ap.add_argument("-is", "--iso", type=int, default=800, help="iso camera sensitivity. higher requires less light")
    ap.add_argument("-b", "--brightness_chg", type=int, default=1.2, help="brightness boost twilight")
    ap.add_argument("-c", "--contrast_chg", type=float, default=1.0, help="contrast boost")
    ap.add_argument("-cl", "--color_chg", type=float, default=1.0, help="color boost")
    ap.add_argument("-sp", "--sharpness_chg", type=float, default=1.0, help="sharpness")

    # prediction defaults
    ap.add_argument("-sc", "--species_confidence", type=float, default=.90,
                    help="species confidence threshold")
    ap.add_argument("-bc", "--bird_confidence", type=float, default=.6, help="bird confidence threshold")
    ap.add_argument("-ma", "--minentropy", type=float, default=5.0,
                    help="min change from first img to current to trigger motion")
    ap.add_argument("-ms", "--minimgperc", type=float, default=10.0,
                    help="ignore objects that are less then % of img")
    ap.add_argument("-hd", "--homedir", type=str, default='/home/pi/birdclassifier/',
                    help="home directory for files")
    ap.add_argument("-la", "--labels", type=str, default='coral.ai.inat_bird_labels.txt',
                    help="name of file to use for species labels and thresholds")
    ap.add_argument("-tr", "--thresholds", type=str, default='USA_WI_coral.ai.inat_bird_threshold.csv',
                    help="name of file to use for species labels and thresholds")
    ap.add_argument("-cm", "--classifier", type=str,
                    default='coral.ai.mobilenet_v2_1.0_224_inat_bird_quant.tflite',
                    help="model name for species classifier")

    # feeder defaults
    ap.add_argument("-ct", "--city", type=str, default='Madison,WI,USA',
                    help="name of city weather station uses OWM web service.  See their site for city options")
    ap.add_argument('-fi', "--feeder_id", type=str, default=hex(uuid.getnode()),
                    help='feeder id default MAC address')
    ap.add_argument('-t', "--feeder_max_temp_c", type=int, default=86,
                    help="Max operating temp for the feeder in C")
    arguments = ap.parse_args()
    if arguments.config_file:
        config = configparser.ConfigParser()
        config.read(arguments.config_file)
        defaults = {}
        defaults.update(dict(config.items()))
        ap.set_defaults(**defaults)
        arguments = ap.parse_args()  # Overwrite arguments with config file
    return arguments


class BirdFeederDetector:
    """
    bird feeder detector class
    """
    def __init__(self, args) -> None:
        """
            initialize bird feeder detector, takes a list of arguments from the command line or a file
            :return: None
            """
        self.args = args
        self.favorite_birds = ['Rose-breasted Grosbeak', 'Red-bellied Woodpecker', 'White-breasted Nuthatch']  # musts
        self.birdpop = population.Census()  # initialize species population census object
        self.gcs_storage = gcs.Storage(offline=self.args.offline)  # init access to google cloud storage
        self.output = output_stream.Controller(caller_id=args.city, gcs_obj=self.gcs_storage,
                                          debug=self.args.debug)  # handle terminal and web output
        self.output.start_stream()  # start streaming to terminal and web
        self.motioncnt = 0
        self.event_count = 0
        self.last_seed_check_hour = 0
        self.gcs_img_filename = ''
        self.seed_check_filename = 'seed_check.jpg'
        self.curr_day = datetime.now().day
        self.curr_hr= datetime.now().hour
        self.last_tweet = datetime(2021, 1, 1, 0, 0, 0)
        self.cityweather = None
        self.motion_detect = None
        self.bird_tweeter = None
        self.chores = None
        self.birds = None
        self.bird_gif = None
        self.local_img_filename = ''
        self.first_img_jpg = None
        self.bird_first_time_seen = False
        return

    def wait_for_sunup(self) -> None:
        """
        patiently wait for the sun to come up before firing up the camera, setup weather object to get current days
        sunrise and sunset
        :return: None
        """
        self.cityweather = weather.CityWeather(city=self.args.city, units='Imperial', iscloudy=60, offline=False)
        self.output.message(
            message=f'Now: {datetime.now()}.  \nSunrise: {self.cityweather.sunrise} Sunset: {self.cityweather.sunset}.',
            msg_type='weather', flush=True)
        self.cityweather.wait_until_midnight()  # if after sunset, wait here until after midnight
        self.cityweather.wait_until_sunrise()  # if before sun rise, wait here
        return

    def setup_camera(self) -> None:
        """
        sun up, start setting up camera and motion detection
        initial video capture, screen size, and grab first image (no motion)
        :return: None
        """
        try:
            self.motion_detect = (
                motion_detector.MotionDetector(min_entropy=self.args.minentropy, screenwidth=self.args.screenwidth,
                                               screenheight=self.args.screenheight, flip_camera=self.args.flipcamera,
                                               first_img_name='first_img.jpg'))
            self.output.message('Done with camera init... setting up classes.')
        except Exception as e:
            print(e)
            self.output.message(message='Camera init failed, check the ribbon', msg_type='message', flush=True)
            time.sleep(60)  # wait for thread to write contents to website
            raise ValueError('List out of range due to camera init failure')
        self.bird_tweeter = tweeter.TweeterClass(offline=self.args.offline)  # init tweeter2 class twitter handler
        self.chores = dailychores.DailyChores(self.bird_tweeter, self.birdpop, self.cityweather,
                                              output_class=self.output)  # setup daily chores obj
        self.birds = det_class_label.DetectClassify(homedir=self.args.homedir, classifier_labels=self.args.labels,
                                                    classifier_model=self.args.classifier,
                                                    classifier_thresholds=self.args.thresholds,
                                                    detect_object_min_confidence=self.args.bird_confidence,
                                                    classify_object_min_confidence=self.args.species_confidence,
                                                    screenheight=self.args.screenheight,
                                                    screenwidth=self.args.screenwidth,
                                                    color_chg=self.args.color_chg,
                                                    contrast_chg=self.args.contrast_chg,
                                                    sharpness_chg=self.args.sharpness_chg,
                                                    brightness_chg=self.args.brightness_chg,
                                                    min_img_percent=self.args.minimgperc,
                                                    target_object=['bird'], debug=self.args.debug,
                                                    output_class=self.output)  # init detection and classifier object
        self.output.message(f'Using label file: {self.birds.labels}')
        self.output.message(f'Using threshold file: {self.birds.thresholds}')
        self.output.message(f'Using classifier file: {self.birds.classifier_file}')
        self.output.message('Starting while loop until sun set..... ')
        self.bird_gif = animate_gif.BirdGif(motion_detector_cls=self.motion_detect, birds_cls=self.birds,
                                            gcs_storage_cls=self.gcs_storage, brightness_chg=self.args.brightness_chg,
                                            min_animated_frames=self.args.minanimatedframes,
                                            stash=self.args.offline)  # obj to create gifs with motion capture
        return

    @staticmethod
    def tweet_text(label: Union[list, str], confidence: Union[list, float]) -> str:
        """
        grab the best label and confidence, use that to generate a twitter text label
        :param label: list of labels or string contain label
        :param confidence: list of confidences or single float
        :return: tweet label as a string
        """
        # sample url https://www.allaboutbirds.org/guide/Northern_Rough-winged_Swallow/overview
        try:
            label = str(label[0]) if isinstance(label, list) else str(label)  # handle list or individual string
            confidence = float(confidence[0]) if isinstance(confidence, list) else float(confidence)  # list or float
            cname = static_functions.common_name(str(label))
            hypername = cname.replace(' ', '_')
            hyperlink = f'https://www.allaboutbirds.org/guide/{hypername}/overview'
            tweet_label = f'{cname} {confidence * 100:.1f}% {hyperlink}'
        except Exception as e:
            tweet_label = ''
            print(e)
        return tweet_label

    def seed_check(self) -> None:
        """
        capture an image of the feeder for website messages page to see if the feeder is empty
        :return: None
        """
        if self.last_seed_check_hour != datetime.now().hour:
            self.last_seed_check_hour = datetime.now().hour
            _ = self.motion_detect.capture_image_with_file(self.seed_check_filename)
            self.gcs_storage.send_file(name=self.seed_check_filename,
                                       file_loc_name=os.getcwd() + '/assets/' + self.seed_check_filename)  # add path
        return

    def process_tweets(self) -> None:
        """
        handle decision to tweet image and process
        :return: None
        """
        # process tweets, jpg if not min number of frame, gif otherwise.  wait X min * N bird before tweeting
        wait_time = self.birdpop.get_single_census_count(self.bird_gif.best_label) * self.args.tweetdelay / 10
        wait_time = self.args.tweetdelay if wait_time >= self.args.tweetdelay else wait_time
        if (datetime.now() - self.last_tweet).total_seconds() >= wait_time or self.bird_first_time_seen or \
                static_functions.common_name(self.bird_gif.best_label) in self.favorite_birds:
            if self.bird_gif.animated:
                self.output.message(
                    message=f'Spotted {self.bird_gif.best_label} {self.bird_gif.best_confidence * 100:.1f}% '
                            f'at {datetime.now().strftime("%I:%M:%S %P")}', event_num=self.event_count,
                    msg_type='spotted', image_name=self.bird_gif.gcs_gif_filename, flush=True)
                if self.bird_tweeter.post_image_from_file(
                        self.tweet_text(self.bird_gif.best_label, self.bird_gif.best_confidence),
                        self.bird_gif.local_gif_filename):
                    self.last_tweet = datetime.now()  # update last tweet time if successful gif posting, ignore fail
            else:
                self.output.message(message=f'Uncertain about a {self.bird_gif.best_label} '
                                            f'{self.bird_gif.best_confidence * 100:.1f}% '
                                            f' with {self.bird_gif.frames_with_birds} frames with birds '
                                            f'at {datetime.now().strftime("%I:%M:%S %P")}',
                                    msg_type='inconclusive', event_num=self.event_count)
        return

    def classify_and_label(self) -> None:
        """
        classify species, grab labels, output census, send to web and terminal,
        enhance the shot, and add boxes, grab next set of gifs, build animation, tweet
        :return: None
        """
        first_rects, first_label, first_conf = self.birds.get_obj_data()  # grab data from this bird
        max_index = self.birds.classified_confidences.index(max(self.birds.classified_confidences))
        file_name = static_functions.common_name(self.birds.classified_labels[max_index]).replace(" ", "")
        gcs_img_filename = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}{str(self.event_count)}' \
                           f'({file_name}).jpg'
        self.output.message(message=f'Possible sighting of a {self.birds.classified_labels[max_index]} '
                                    f'{self.birds.classified_confidences[max_index] * 100:.1f}% at '
                                    f'{datetime.now().strftime("%I:%M:%S %P")}', event_num=self.event_count,
                            msg_type='possible', image_name=gcs_img_filename, flush=True)  # 2 web stream
        # check for need of final image adjustments
        self.first_img_jpg = self.first_img_jpg if self.args.brightness_chg == 0 \
                                                   or self.cityweather.isclear or self.cityweather.is_twilight() \
            else image_proc.enhance(img=self.first_img_jpg, brightness=self.args.brightness_chg)
        first_img_jpg_no_label = self.first_img_jpg.copy()

        # create animation: unlabeled first image is passed to gif function, bare copy is annotated later
        _gif = self.bird_gif.build_gif(self.event_count, self.first_img_jpg)

        # annotate bare image copy, use either best gif label or org data
        best_first_label = (
            static_functions.convert_to_list(
                self.bird_gif.best_label if self.bird_gif.best_label != '' else first_label))
        best_first_conf = (
            static_functions.convert_to_list(self.bird_gif.best_confidence if self.bird_gif.best_confidence > 0 else
                                             first_conf))
        self.bird_first_time_seen = self.birdpop.record_visitor(best_first_label, datetime.now())  # inc species count
        self.birds.set_ojb_data(classified_rects=first_rects, classified_labels=best_first_label,
                                classified_confidences=best_first_conf)  # set to first bird
        first_img_jpg = self.birds.add_boxes_and_labels(label_img=first_img_jpg_no_label, use_last_known=False)
        first_img_jpg.save(self.local_img_filename)
        self.gcs_storage.send_file(name=gcs_img_filename, file_loc_name=self.local_img_filename)

        # save unlabeled jpg, wrap in try except since this is untested as yet.
        try:
            raw_file_name = os.getcwd() + '/assets/raw_' + str(self.event_count % 10) + '.jpg'
            raw_gcs_img_filename = f'raw_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}{str(self.event_count)}' \
                           f'({file_name} {str(first_rects)}).jpg'
            first_img_jpg_no_label.save(raw_file_name)
            self.gcs_storage.send_file(name=raw_gcs_img_filename, file_loc_name=raw_file_name)
        except Exception as e:
            print(e)
            print('Error in saving unlabeled jpg')
            pass
        return

    def detect_birds(self) -> None:
        """
        run wait for sunup and setup camera first, this method starts the entire process until sundown
        :return: None
        """
        self.wait_for_sunup()
        self.setup_camera()
        # loop while the sun is up, look for motion, detect birds, determine species
        while self.cityweather.sunrise.time() < datetime.now().time() < self.cityweather.sunset.time():
            self.seed_check()
            self.chores.hourly_and_daily(filename='seed_check.jpg')  # weather reporting, cpu checks, seed check img
            self.motion_detect.detect()

            # check if day time, is motion, is a bird, and image not overexposed
            if self.motion_detect.motion and self.birds.detect(detect_img=self.motion_detect.img) and \
                    self.cityweather.is_dawn() is False and self.cityweather.is_dusk() is False \
                    and image_proc.is_sun_reflection_jpg(img=self.motion_detect.img, debug=self.args.debug) is False:
                self.motion_detect.reset_motion_count()  # used for debugging in motion_detect object
                self.birds.set_colors()  # set new colors for this series of bounding boxes
                self.event_count += 1  # increment event code for log and messages
                self.local_img_filename = os.getcwd() + '/assets/' + str(self.event_count % 10) + '.jpg'
                self.first_img_jpg = self.birds.img  # keep first shot for animation and web
                if self.birds.classify(class_img=self.first_img_jpg) >= self.args.species_confidence:  # classify bird
                    self.classify_and_label()  # found classifications, set correct bird, and label image accordingly
                    self.process_tweets()  # handle decision to tweet and related steps

        self.output.end_stream()  # ending process for evening, print blank line and shut down
        self.motion_detect.stop()
        if self.args.verbose:
            self.chores.hourly_and_daily(report_pop=True)
            self.chores.end_report()  # post a report on run time of the process
        return


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    args_construct = process_args()
    bird_feeder = BirdFeederDetector(args_construct)
    bird_feeder.detect_birds()
    # bird_detector(args_construct)  # old code

