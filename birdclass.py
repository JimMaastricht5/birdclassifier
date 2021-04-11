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
# motion detector with builtin bird detection and bird classification
# built by JimMaastricht5@gmail.com
# uses tflite prebuilt google model for object detection and tensorflow lite model for bird classification
# currently using a model from coral.ai for species identification.
# Other species models built with tensor flow  using tools from tensor in colab notebook
# https://colab.research.google.com/drive/1taZ9JincTaZuZh_JmBSC4pAbSQavxbq5#scrollTo=D3i_6WSXjUhk
# special notes when setting up the code on a rasberry pi 4 9/9/20
# install supporting libraries for directions @: https://qengineering.eu/install-opencv-4.2-on-raspberry-pi-4.html
# install OpenCv version 4.4+
# packages: twitter use twython package, auth.py must be in project for import auth
#   oauthlib,
import cv2  # open cv 2
import label_image  # code to init tensor flow model and classify bird type
import motion_detector  # motion detector helper functions
import weather
import tweeter  # twitter helper functions
import image_proc  # lib of image enhancement functions
import population  # population census object, tracks species total seen and last time
import argparse  # argument parser
import numpy as np
from datetime import datetime
import logging


def bird_detector(args):
    colors = np.random.uniform(0, 255, size=(11, 3))  # random colors for bounding boxes
    birdpop = population.Census()  # initialize species population census object
    motioncnt = 0
    curr_day, curr_hr = datetime.now().day, datetime.now().hour
    spweather = weather.City_Weather()  # init class and set var based on default of Madison WI

    # initial video capture, screen size, and grab first image (no motion)
    cap = cv2.VideoCapture(0)  # capture video image
    cap.set(3, args["screenwidth"])  # set screen width
    cap.set(4, args["screenheight"])  # set screen height
    first_img = motion_detector.init(args["flipcamera"], cv2, cap)  # set gray motion mask
    set_windows()  # position output windows at top of screen and init output

    # load species threshold file, note this will not handles species as a string in the first column.
    species_thresholds = np.genfromtxt(args["species_thresholds"], delimiter=',')

    # setup twitter and tensor flow models
    bird_tweeter = tweeter.Tweeter_Class()  # init tweeter2 class twitter handler
    tfobjdet, objdet_possible_labels = label_image.init_tf2(args["obj_det_model"], args["numthreads"],
                                                            args["obj_det_labels"])
    interpreter, possible_labels = label_image.init_tf2(args["species_model"], args["numthreads"],
                                                        args["species_labels"])

    bird_tweeter.post_status(f'Starting process at {datetime.now().strftime("%I:%M:%S %P")}, ' +
                            f'{spweather.weatherdescription} ' +
                            f'with {spweather.skycondition}% cloud cover amd visibility of {spweather.visibility} ft.' +
                            f' Temp is currently {spweather.temp}F.' +
                            f'Wind speeds of {spweather.windspeed}F.')

    while True:  # while escape key is not pressed look for motion, detect birds, and determine species
        species_conf = 0  # init species confidence
        curr_day, curr_hr = hour_or_day_change(curr_day, curr_hr, spweather, bird_tweeter, birdpop)

        motionb, img = motion_detector.detect(args["flipcamera"], cv2, cap, first_img, args["minarea"])
        if motionb:  # motion detected.
            motioncnt += 1
            print(f'\r motion {motioncnt}', end=' ')  # indicate motion on monitor

            # improve image prior to obj detection and labeling
            if spweather.isclear is False or image_proc.is_color_low_contrast(img):
                equalizedimg = image_proc.equalize_color(img)  # balance histogram of color intensity
            else:
                equalizedimg = img.copy()  # no adjustment necessary, create a copy of the image

            det_confidences, det_labels, det_rects = \
                label_image.object_detection(args["bconfidence"], img, objdet_possible_labels, tfobjdet,
                                             args["inputmean"], args["inputstd"])  # detect objects

            for i, det_confidence in enumerate(det_confidences):  # loop thru detected objects
                loginfo = f"{det_labels[i]}:{det_confidence * 100:.0f}%"
                logging.info(datetime.now().strftime('%I:%M:%S %p') + loginfo)
                print(': ' + datetime.now().strftime('%I:%M %p') + ' observed ' + loginfo, end='')

                if det_labels[i] == "bird":  # bird observed, find species, label, and tweet
                    motioncnt = 0  # reset motion count between birds
                    (startX, startY, endX, endY) = label_image.scale_rect(img, det_rects[i])  # set x,y bounding box
                    birdcrop_img = equalizedimg[startY:endY, startX:endX]  # extract image for better species detection
                    species_conf, species = label_image.set_label(birdcrop_img, possible_labels, species_thresholds,
                                                                  interpreter, args["inputmean"], args["inputstd"])
                    birdpop.visitor(species, datetime.now())  # update census
                    common_name, img_label, tweet_label = label_text(species, species_conf)
                    img = label_image.add_box_and_label(img, '', startX, startY, endX, endY, colors, i)  # add box 2 vid
                    equalizedimg = label_image.add_box_and_label(equalizedimg, img_label, startX, startY,
                                                                 endX, endY, colors, i)  # add box and label
                    cv2.imshow('equalized', equalizedimg)

            # all birds in image processed. Show image and tweet, confidence here is lowest in the picture
            if species_conf >= args["sconfidence"]:  # tweet threshold
                species_count, species_last_seen = birdpop.report_census(species)  # get census
                if (species_last_seen - datetime.now()).total_seconds() >= 60 * 5:
                    if bird_tweeter.post_image(tweet_label + str(species_count + 1), equalizedimg):
                        cv2.imshow('tweeted', equalizedimg)  # show tweeted picture with labels
                    else:
                        print(f" {species} seen {species_last_seen.strftime('%I:%M %p')} *** exceeded tweet limit")
                else:
                    print(f" {species} not tweeted, last seen {species_last_seen.strftime('%I:%M %p')}. wait 5 minutes")

        cv2.imshow('video', img)  # show image with box and label use cv2.flip if image inverted
        cv2.waitKey(20)  # wait 20 ms to render video, restart loop.  setting of 0 is fixed img; > 0 video
        if curr_hr == 20:  # is it 10pm, if so shut down.  cron start at 0 5 * * birdclass.sh
            break

    # while loop break at 10pm, shut down windows
    cap.release()
    cv2.destroyAllWindows()
    bird_tweeter.post_status(f'Ending process at {datetime.now().strftime("%I:%M:%S %P")}.')


# housekeeping for day and hour
def hour_or_day_change(curr_day, curr_hr, spweather, bird_tweeter, birdpop):
    if curr_day != datetime.now().day:
        observed = birdpop.get_census_by_count()  # count from prior day
        bird_tweeter.post_status(f'top 3 birds for day {str(curr_day)}')
        index, loopcnt = 0, 1
        while loopcnt <= 3:  # print top 3 skipping unknown species
            if observed[index][0:2] == '':
                index += 1
            try:
                bird_tweeter.post_status(f'#{str(loopcnt)} {observed[index][0:2]}')
            except:
                bird_tweeter.post_status('unable to post observations')
                break
            index += 1
            loopcnt += 1

        birdpop.clear()  # clear count for new day
        curr_day = datetime.now().day  # set new day = to current day

    if curr_hr != datetime.now().hour:  # check weather pattern hourly
        spweather.update_conditions()
        curr_hr = datetime.now().hour
    return curr_day, curr_hr


# set label for image and tweet, use short species name instead of scientific name
def label_text(species, species_conf):
    species = str(species)  # make sure species is considered a string
    start = species.find('(') + 1  # find start of common name, move one character to drop (
    end = species.find(')')
    if start >= 0 and end >= 0:
        common_name = species[start:end]
    else:
        common_name = species
    tweet_label = f"{species}: confidence {species_conf * 100:.0f} observed: "
    return common_name, common_name, tweet_label


def set_windows():
    cv2.namedWindow('video')
    cv2.namedWindow('equalized')
    cv2.namedWindow('tweeted')

    cv2.moveWindow('video', 0, 0)
    cv2.moveWindow('equalized', 350, 0)
    cv2.moveWindow('tweeted', 700, 0)

    # cv2.imshow('video', img)
    # cv2.imshow('equalized', img)
    # cv2.imshow('tweeted', img)
    # cv2.waitKey(20)  # wait 20 ms to render video, restart loop.  setting of 0 is fixed img; > 0 video
    return


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--flipcamera", type=bool, default=False, help="flip camera image")
    ap.add_argument("-a", "--minarea", type=int, default=1000, help="motion threshold")
    ap.add_argument("-sw", "--screenwidth", type=int, default=320, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=240, help="max screen height")

    # object detection model setup
    ap.add_argument('-om', "--obj_det_model",
                    default='/home/pi/birdclass/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite')
    ap.add_argument('-p', '--obj_det_labels',
                    default='/home/pi/PycharmProjects/pyface2/lite-model_ssd_mobilenet_v1_1_metadata_2_labelmap.txt')

    # species model setup
    ap.add_argument('-m', '--species_model',
                    default='/home/pi/PycharmProjects/pyface2/coral.ai.mobilenet_v2_1.0_224_inat_bird_quant.tflite',
                    help='.tflite model to be executed')
    ap.add_argument('-l', '--species_labels',
                    default='/home/pi/PycharmProjects/pyface2/coral.ai.inat_bird_labels.txt',
                    help='name of file containing labels')
    ap.add_argument('-ts', '--species_thresholds',
                    default='/home/pi/PycharmProjects/pyface2/coral.ai.inat_bird_threshold.csv',
                    help='name of file containing thresholds by label')

    # tensor flow input arguements
    ap.add_argument('--inputmean', default=127.5, type=float, help='Tensor input_mean')
    ap.add_argument('--inputstd', default=127.5, type=float, help='Tensor input standard deviation')
    ap.add_argument('--numthreads', default=None, type=int, help='Tensor number of threads, leave at default')

    # confidence settings for object detection and species bconfidence
    ap.add_argument('-bc', '--bconfidence', type=float, default=0.65)  # obj detection threshold; 76 is a good min
    ap.add_argument('-sc', '--sconfidence', type=float, default=0.90)  # quant model is accurate down to 90

    logging.basicConfig(filename='birdclass.log', format='%(asctime)s - %(message)s', level=logging.DEBUG)
    arguments = vars(ap.parse_args())
    bird_detector(arguments)
