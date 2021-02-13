# motion detector with builtin bird detection and bird classification
# built by JimMaastricht5@gmail.com
# uses Haar Cascade for bird detection and tensorflow lite model for bird classification
# tensor flow model built using tools from tensor in colab notebook
# https://colab.research.google.com/drive/1taZ9JincTaZuZh_JmBSC4pAbSQavxbq5#scrollTo=D3i_6WSXjUhk
# model trained to find birds common to feeders in WI using caltech bird dataset 201
# special notes when setting up the code on a rasberry pi 4 9/9/20
# install supporting libraries for directions @: https://qengineering.eu/install-opencv-4.2-on-raspberry-pi-4.html
# install OpenCv version 4.4+
# packages: twitter use twython package, auth.py must be in project for import auth
#   oauthlib,
import cv2  # open cv 2
import label_image  # code to init tensor flow model and classify bird type
import motion_detector  # motion detector helper functions
import tweeter  # twitter helper functions
import image_proc  # lib of image enhancement functions
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
    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    colors = np.random.uniform(0, 255, size=(11, 3))  # random colors for bounding boxes
    starttime = datetime(2021, 1, 1, 0, 0, 0, 0)  # init start time for observation delay

    # initial video capture, screen size, and grab first image (no motion)
    cap = cv2.VideoCapture(0)  # capture video image
    cap.set(3, args["screenwidth"])  # set screen width
    cap.set(4, args["screenheight"])  # set screen height
    first_img = motion_detector.init(cv2, cap)
    # init twitter and tf lite obj detection and species model file
    twitter = tweeter.init(api_key, api_secret_key, access_token, access_token_secret)  # init twitter api
    tfobjdet, objdet_possible_labels = label_image.init_tf2(args["obj_det_model"], args["numthreads"],
                                                            args["obj_det_labels"])
    interpreter, possible_labels = label_image.init_tf2(args["species_model"], args["numthreads"],
                                                        args["species_labels"])
    print('press esc to quit')
    # main loop ******
    while True:  # while escape key is not pressed
        birdb = False
        tweetb = False
        img_label = ''
        motionb, img, gray, graymotion, thresh = motion_detector.detect(cv2, cap, first_img, args["minarea"])
        if motionb:  # motion detected.
            det_confidences, det_labels, det_rects = label_image.object_detection(args["bconfidence"], img,
                                                                                  objdet_possible_labels, tfobjdet,
                                                                                  args["inputmean"], args["inputstd"])
            for i, det_confidence in enumerate(det_confidences):
                logtime = datetime.now().strftime('%H:%M:%S')
                loginfo = "---saw  {}: {:.2f}%".format(det_labels[i], det_confidence * 100)
                logging.info(logtime + loginfo)
                print(loginfo, logtime)
                if image_proc.is_low_contrast(img):
                    print('low contrast image')

                if det_labels[i] == "bird" and not image_proc.is_low_contrast(img) \
                        and (det_confidence >= args["bconfidence"] or birdb):
                    birdb = True  # set to loop thru img for other birds in pic
                    (startX, startY, endX, endY) = label_image.scale_rect(img, det_rects[i])  # set x,y bounding box

                    # *** need to improve bird size measurements w/known sized obj in pic, depth is big issue
                    bird_size, bird_per_scr_area = birdsize(args, startX, startY, endX, endY)  # determine bird size

                    # extract image of bird, use crop for better species detection in models
                    # *** need to color calobrate for species model
                    equalizedimg = image_proc.equalize_color(img)
                    birdcrop_img = equalizedimg[startY:endY, startX:endX]
                    color = label_image.predominant_color(birdcrop_img)  # find main color of bird
                    species_conf, species = label_image.set_label(birdcrop_img, possible_labels, interpreter,
                                                                  args["inputmean"], args["inputstd"])

                    # draw bounding boxes and display label if it is a bird
                    tweetb, img_label = set_img_label(args, det_confidence, species, species_conf, bird_size,
                                                      bird_per_scr_area, color)

                    # add bounding boxes and labels to images
                    img = label_image.add_box_and_label(img, img_label, startX, startY, endX, endY, colors, i)
                    equalizedimg = label_image.add_box_and_label(equalizedimg, img_label, startX, startY,
                                                                 endX, endY, colors, i)

            if birdb:  # if object detection saw a bird draw the results
                cv2.imshow('org detection', img)  # show all birds in pic with labels
                cv2.imshow('color histogram equalized', equalizedimg)

            # image contained a bird and species label, tweet it
            if tweetb and (datetime.now() - starttime).total_seconds() > 1800:  # wait 30 min in seconds
                starttime = datetime.now()
                logdate = starttime.strftime('%H:%M:%S')
                logging.info('*** tweeted ' + logdate + ' ' + img_label)
                print('*** tweeted', logdate, img_label)
                cv2.imshow('tweeted', equalizedimg)  # show all birds in pic with labels
                cv2.imwrite("img.jpg", equalizedimg)  # write out image for debugging and testing
                tw_img = open('img.jpg', 'rb')
                tweeter.post_image(twitter, img_label, tw_img)
            elif tweetb:
                print('--- wait 30 min for next tweet:{:.2f}'.format(
                    (datetime.now().timestamp() - starttime.timestamp()) / 60))

        ret, videoimg = cap.read()
        # videoimg = cv2.flip(videoimg, -1)  # mirror image; comment out if not needed for your camera
        cv2.imshow('video', videoimg)

        # check for esc key and quit if pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break

    if args["image"] == "":  # not testing code
        cap.release()
    cv2.destroyAllWindows()


# estimate the size of the bird based on the percentage of image area consumed by the bounding box
def birdsize(args, startx, starty, endx, endy):
    birdarea = abs((startx - endx) * (starty - endy))
    scrarea = args['screenheight'] * args['screenwidth']
    perarea = (birdarea / scrarea) * 100
    if perarea >= 40:  # large bird
        size = 'L'
    elif perarea >= 30:  # medium bird
        size = 'M'
    else:  # small bird usually ~ 20%
        size = 'S'
    logging.info(str(birdarea) + ' ' + str(scrarea) + ' ' + str(perarea) + ' ' + size)
    return size, perarea


# set label for image and tweet, use short species name instead of scientific name
def set_img_label(args, bird_conf, species, species_conf, bird_size, bird_per_scr_area, color):
    if species_conf < args["sconfidence"]:  # low confidence in species
        print('--- low {} confidence {.2f}, bird {.2f}'.format(species, species_conf * 100, bird_conf * 100))
        species = 'bird'  # reset species to bird due to low confidence
    start = species.find('(')
    end = species.find(')')
    if start >= 0 and end >= 0:
        common_name = species[start:end]
    else:
        common_name = species
    img_label = "{}: {:.2f}".format(common_name, species_conf * 100)
    logging.info('--- ' + img_label + ' ' + bird_size + ' ' + ' ' + color + ' ' + str(bird_per_scr_area))  # log info
    print('--- ' + img_label + ' ' + bird_size + ' ' + ' ' + color + ' ' + str(bird_per_scr_area))  # display to term
    return (species_conf >= args["sconfidence"]), img_label


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # video setup
    # ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--minarea", type=int, default=2, help="motion threshold, lower triggers more often")
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

    # tensor flow input arguements
    ap.add_argument('--inputmean', default=127.5, type=float, help='Tensor input_mean')
    ap.add_argument('--inputstd', default=127.5, type=float, help='Tensor input standard deviation')
    ap.add_argument('--numthreads', default=None, type=int, help='Tensor number of threads')

    # confidence settings for object detection and species bconfidence
    ap.add_argument('-bc', '--bconfidence', type=float, default=0.80)
    ap.add_argument('-sc', '--sconfidence', type=float, default=0.95)

    logging.basicConfig(filename='birdclass.log', format='%(asctime)s - %(message)s', level=logging.INFO)
    arguments = vars(ap.parse_args())
    print(datetime.now().strftime('%H:%M:%S'))
    bird_detector(arguments)
