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
from auth import (
    api_key,
    api_secret_key,
    access_token,
    access_token_secret
)  # import twitter keys
import logging


def bird_detector(args):
    colors = np.random.uniform(0, 255, size=(11, 3))  # random colors for bounding boxes
    birdpop = population.Census()  # initialize species population census object
    motioncnt, tweetcnt = 0, 0
    curr_day, curr_hr = datetime.now().day, datetime.now().hour

    # initial video capture, screen size, and grab first image (no motion)
    cap = cv2.VideoCapture(0)  # capture video image
    cap.set(3, args["screenwidth"])  # set screen width
    cap.set(4, args["screenheight"])  # set screen height
    first_img = motion_detector.init(args["flipcamera"], cv2, cap)  # set gray motion mask

    # load species threshold file, note this will not handles species as a string in the first column.
    species_thresholds = np.genfromtxt(args["species_thresholds"], delimiter=',')

    # init twitter and tensor flow models
    twitter = tweeter.init(api_key, api_secret_key, access_token, access_token_secret)  # init twitter api
    tfobjdet, objdet_possible_labels = label_image.init_tf2(args["obj_det_model"], args["numthreads"],
                                                            args["obj_det_labels"])
    interpreter, possible_labels = label_image.init_tf2(args["species_model"], args["numthreads"],
                                                        args["species_labels"])
    print('press esc to quit')
    while True:  # while escape key is not pressed look for motion, detect birds, and determine species
        species_conf = 0  # set species confidence to zero for next loop
        if curr_day != datetime.now().day:
            observed = birdpop.get_census_by_count()  # print count from prior day
            try:
                tweeter.post_status(twitter, f'top 3 birds for day {str(curr_day)}: #1 {observed[0][0:2]}')
                tweeter.post_status(twitter, f'#2 {observed[1][0:2]}')
                tweeter.post_status(twitter, f'#3 {observed[2][0:2]}')
            except:
                tweeter.post_status(twitter, 'unable to post observations')
            birdpop.clear()  # clear count for new day
            curr_day = datetime.now().day

        if curr_hr != datetime.now().hour:
            tweetcnt = 0  # reset hourly twitter limit

        motionb, img = motion_detector.detect(args["flipcamera"], cv2, cap, first_img, args["minarea"])
        if motionb:  # motion detected.
            motioncnt += 1
            print(f'\r motion {motioncnt}', end=' ')  # indicate motion on monitor

            # detect objects in image and loop thru them
            det_confidences, det_labels, det_rects = \
                label_image.object_detection(args["bconfidence"], img, objdet_possible_labels, tfobjdet,
                                             args["inputmean"], args["inputstd"])

            for i, det_confidence in enumerate(det_confidences):
                loginfo = f"{det_labels[i]}:{det_confidence * 100:.0f}%"
                logging.info(datetime.now().strftime('%H:%M:%S') + loginfo)
                print(':' + loginfo, datetime.now().strftime('%H:%M'), end='')

                # bird observed, determine species, label images, increment population observation and tweet
                if det_labels[i] == "bird" and not image_proc.is_low_contrast(img) \
                        and (det_confidence >= args["bconfidence"]):
                    motioncnt = 0
                    (startX, startY, endX, endY) = label_image.scale_rect(img, det_rects[i])  # set x,y bounding box
                    if weather.is_cloudy():
                        equalizedimg = image_proc.equalize_color(img)  # balance histogram of color intensity
                    else:
                        equalizedimg = img  # no adjustment necessary

                    birdcrop_img = equalizedimg[startY:endY, startX:endX]  # extract image for better species detection
                    species_conf, species = label_image.set_label(birdcrop_img, possible_labels, species_thresholds,
                                                                  interpreter, args["inputmean"], args["inputstd"])
                    species_count, species_last_seen = birdpop.report_census(species)
                    # draw bounding boxes and display label if it is a bird
                    common_name, img_label, tweet_label = label_text(species, species_conf)
                    orgimg = label_image.add_box_and_label(img, img_label, startX, startY, endX, endY, colors, i)
                    img = label_image.add_box_and_label(img, '', startX, startY, endX, endY, colors, i)  # add box 2 vid
                    equalizedimg = label_image.add_box_and_label(equalizedimg, img_label, startX, startY,
                                                                 endX, endY, colors, i)

                    print(f'\n best fit: {img_label} {(species_conf * 100)} observed: {str(species_count + 1)}')
                    cv2.imshow('org detection', orgimg)  # show all birds in pic with labels
                    cv2.imshow('color histogram equalized', equalizedimg)

            # all birds in image processed. Show image and tweet, confidence here is lowest across all species
            if species_conf >= args["sconfidence"]:
                if tweetcnt < 100:  # no more than 100 per hour
                    # species_count, species_last_seen = birdpop.report_census(species)  # needed?
                    cv2.imshow('tweeted', equalizedimg)  # show all birds in pic with labels
                    cv2.imwrite("img.jpg", equalizedimg)  # write out image for debugging and testing
                    tw_img = open('img.jpg', 'rb')  # reload a image for twitter, correct var type
                    tweeter.post_image(twitter, tweet_label + str(species_count + 1), tw_img)
                    tweetcnt += 1
                else:
                    print(f" {species} seen {species_last_seen.strftime('%H:%M')} *** exceeded tweet limit")
                birdpop.visitor(species, datetime.now())  # update visitor count

        cv2.imshow('video', img)  # show image with box and label use cv2.flip if image inverted
        k = cv2.waitKey(30)  # wait 30 milliseconds for key press
        if k == 27:  # press 'ESC' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


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


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--flipcamera", type=bool, default=False, help="flip camera image")
    ap.add_argument("-a", "--minarea", type=int, default=35000, help="motion threshold, 35k to 50K is a good min")
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
    ap.add_argument('-bc', '--bconfidence', type=float, default=0.76)  # obj detection threshold; 76 is a good min
    ap.add_argument('-sc', '--sconfidence', type=float, default=0.90)  # quant model is accurate down to 90

    logging.basicConfig(filename='birdclass.log', format='%(asctime)s - %(message)s', level=logging.DEBUG)
    arguments = vars(ap.parse_args())
    print(datetime.now().strftime('%H:%M:%S'))
    bird_detector(arguments)
