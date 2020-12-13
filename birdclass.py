# motion detector with builtin bird detection and bird classification
# built by JimMaastricht5@gmail.com
# uses Haar Cascade for bird detection and tensorflow lite model for bird classification
# tensor flow model built using tools from tensor in colab notebook
# https://colab.research.google.com/drive/1taZ9JincTaZuZh_JmBSC4pAbSQavxbq5#scrollTo=D3i_6WSXjUhk
# model trained to find birds common to feeders in WI using caltech bird dataset 2011
# 017.Cardinal
# 049.Boat_tailed_Grackle
# 094.White_breasted_Nuthatch
# 118.House_Sparrow
# 129.Song_Sparrow
# 300.American_Goldfinch
# 301.Baltimore_Oriole
# 302.Black-Capped_Chickadee
# 303.Brown_Thrasher
# 305.Common_Starling
# 306.Dark_Eyed_Junco
# 307.Eastern_Bluebird
# 308.Eastern_Meadowlark
# 309.Eastern_Towee
# 310.Evening_Grosbeak
# 312.House_Finch
# 316.Purple_Finch
# 317.Robin
# 318.Tit_Mouse


# special notes when setting up the code on a rasberry pi 4 9/9/20
# install supporting libraries for directions @: https://qengineering.eu/install-opencv-4.2-on-raspberry-pi-4.html
# install OpenCv version 4.4+
# packages: pan tilt uses PCA9685-Driver, twitter use twython package, auth.py must be in project for import auth
#   oauthlib,
import cv2  # open cv 2
import label_image  # code to init tensor flow model and classify bird type
import PanTilt9685  # pan tilt control code
import motion_detector  # motion detector helper functions
import tweeter  # twitter helper functions
import argparse  # argument parser
import numpy as np
import time
from datetime import datetime
from time import strftime
from auth import (
    api_key,
    api_secret_key,
    access_token,
    access_token_secret
)  # import twitter keys
import logging


def bird_detector(args):
    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    colors = np.random.uniform(0, 255, size=(11, 3))  # random colors for bounding boxes
    birds_found = []
    starttime = time.time()

    # setup pan tilt and initialize variables
    if args["panb"]:
        currpan, currtilt, pwm = PanTilt9685.init_pantilt()

    # initial video capture, screen size, and grab first image (no motion)
    if args["image"] == "":
        try:
            cap = cv2.VideoCapture(0)  # capture video image
        except:
            print('camera failure on initial video capture, ending program')
            logging.critical('camera failure on initial video capture, ending program')
            return

        cap.set(3, args["screenwidth"])  # set screen width
        cap.set(4, args["screenheight"])  # set screen height
        first_img = motion_detector.init(cv2, cap)
    else:
        first_img = cv2.imread(args["image"])  # testing code

    twitter = tweeter.init(api_key, api_secret_key, access_token, access_token_secret)  # init twitter api

    # init tf lite obj detection and species model file
    tfobjdet, objdet_possible_labels = label_image.init_tf2(args["obj_det_model"], args["numthreads"],
                                                            args["obj_det_labels"])
    interpreter, possible_labels = label_image.init_tf2(args["species_model"], args["numthreads"],
                                                        args["species_labels"])

    print('press esc to quit')

    # main loop ******
    while True:  # while escape key is not pressed
        if args["image"] == "":
            try:
                motionb, img, gray, graymotion, thresh = motion_detector.detect(cv2, cap, first_img, args["minarea"])
            except:
                logging.critical('camera failed on motion capture and compare of second image, ending program')
                return

        else:  # testing code
            motionb = True
            first_img = cv2.imread(args["image"])
            img = cv2.imread(args["image"])

        if motionb:  # motion detected.
            det_confidences, det_labels, det_rects = label_image.object_detection(args["confidence"], img,
                                                                                  objdet_possible_labels, tfobjdet,
                                                                                  args["inputmean"], args["inputstd"])
            tweetb = False
            combined_label = ''
            for i, det_confidence in enumerate(det_confidences):
                loginfo = datetime.now().strftime('%H:%M:%S')
                loginfo = loginfo + " saw  {}: {:.2f}%".format(det_labels[i], det_confidence * 100)
                logging.info(loginfo)
                print(loginfo)
                if det_labels[i] == "bird" and (det_confidence >= args["confidence"] or tweetb):
                    (startX, startY, endX, endY) = label_image.scale_rect(img, det_rects[i])  # x,y coord bounding box
                    ts_img = img[startY:endY, startX:endX]  # extract image of bird
                    tfconfidence, birdclass = label_image.set_label(ts_img, possible_labels, interpreter,
                                                                    args["inputmean"], args["inputstd"])

                    # draw bounding boxes and display label if it is a bird
                    if tfconfidence >= args["bconfidence"]:  # high confidence in species
                        tweetb = True
                        label = "{}: {:.2f}% bird: {:.2f}%".format(birdclass, tfconfidence * 100, det_confidence * 100)
                    else:
                        loginfo = label = "bird, confidence species {}: {:.2f}% bird: {:.2f}%".format(birdclass,
                                                                            tfconfidence * 100, det_confidence * 100)
                        logging.info(loginfo)
                        print(loginfo)
                        label = "{}: {:.2f}%".format("bird", det_confidence * 100)
                        birdclass = 'bird'

                    combined_label = combined_label + ' ' + label  # build label for multi birds in one photo
                    cv2.rectangle(img, (startX, startY), (endX, endY), colors[i], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15  # adjust label loc if too low
                    cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

                    if birdclass in birds_found:  # seen it
                        loginfo = label + ' last seen at: ' + starttime.strftime('%H:%M:%S')
                        logging.info(loginfo)
                        # print(loginfo)
                        if (time.time() - starttime) >= 300:  # 5 min elapsed time in seconds;
                            starttime = time.time()  # reset timer
                            birds_found = []  # clear birds found
                        else:  # something new is at the feeder
                            birds_found.append(birdclass)

            if tweetb:  # image contained a bird and species label
                cv2.imshow('obj detection', img)  # show all birds in pic with labels
                cv2.imwrite("img.jpg", img)  # write out image for debugging and testing
                tw_img = open('img.jpg', 'rb')
                tweeter.post_image(twitter, combined_label, tw_img)
                birds_found.append(birdclass)

        if args["panb"]:
            currpan, currtilt = PanTilt9685.trackobject(pwm, cv2, currpan, currtilt, img,
                                                        (startX, startY, endX, endY),
                                                        args["screenwidth"], args["screenheight"])
        ret, videoimg = cap.read()
        # videoimg = cv2.flip(videoimg, -1)  # mirror image; comment out if not needed for your camera
        cv2.imshow('video', videoimg)
        # cv2.imshow('gray', graymotion)
        # cv2.imshow('threshold', thresh)

        # check for esc key and quit if pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break

    if args["image"] == "":  # not testing code
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # video setup
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--minarea", type=int, default=5, help="minimum area size")
    ap.add_argument("-sw", "--screenwidth", type=int, default=320, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=240, help="max screen height")

    # object detection model setup
    ap.add_argument('-om', "--obj_det_model",
                    default='/home/pi/birdclass/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite')
    ap.add_argument('-p', '--obj_det_labels',
                    default='/home/pi/PycharmProjects/pyface2/lite-model_ssd_mobilenet_v1_1_metadata_2_labelmap.txt')

    # species model setup
    ap.add_argument('-m', '--species_model',
                    default='/home/pi/PycharmProjects/pyface2/birdskgc-s-224-92.44.tflite',
                    help='.tflite model to be executed')
    ap.add_argument('-l', '--species_labels',
                    default='/home/pi/PycharmProjects/pyface2/birdskgc-17.txt',
                    help='name of file containing labels')

    # tensor flow input arguements
    ap.add_argument('--inputmean', default=127.5, type=float, help='Tensor input_mean')
    ap.add_argument('--inputstd', default=127.5, type=float, help='Tensor input standard deviation')
    ap.add_argument('--numthreads', default=None, type=int, help='Tensor number of threads')

    # confidence settings for object detection and species bconfidence
    ap.add_argument('-c', '--confidence', type=float, default=0.80)
    ap.add_argument('-bc', '--bconfidence', type=float, default=0.975)

    # set pan tilt control model
    ap.add_argument('--panb', default=False, type=bool, help='activate pan tilt mechanism')

    # input test image or use video
    # ap.add_argument('-i', '--image', default='/home/pi/birdclass/cardinal.jpg',
    #                                         help='image to be classified')
    ap.add_argument('-i', '--image', default='', help='image to be classified')

    logging.basicConfig(filename='birdclass.log', format='%(asctime)s - %(message)s', level=logging.INFO)
    arguments = vars(ap.parse_args())
    print(datetime.now().strftime('%H:%M%S'))
    bird_detector(arguments)
