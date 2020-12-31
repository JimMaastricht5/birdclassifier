# motion detector with builtin bird detection and bird classification
# built by JimMaastricht5@gmail.com
# uses Haar Cascade for bird detection and tensorflow lite model for bird classification
# tensor flow model built using tools from tensor in colab notebook
# https://colab.research.google.com/drive/1taZ9JincTaZuZh_JmBSC4pAbSQavxbq5#scrollTo=D3i_6WSXjUhk
# model trained to find birds common to feeders in WI using caltech bird dataset 201
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
    birds_found = []
    starttime = datetime.now()

    # setup pan tilt and initialize variables
    if args["panb"]:
        currpan, currtilt, pwm = PanTilt9685.init_pantilt()

    # initial video capture, screen size, and grab first image (no motion)
    try:
        cap = cv2.VideoCapture(0)  # capture video image
    except:
        print('camera failure on initial video capture, ending program')
        logging.critical('camera failure on initial video capture, ending program')
        return
    cap.set(3, args["screenwidth"])  # set screen width
    cap.set(4, args["screenheight"])  # set screen height
    first_img = motion_detector.init(cv2, cap)

    twitter = tweeter.init(api_key, api_secret_key, access_token, access_token_secret)  # init twitter api

    # init tf lite obj detection and species model file
    tfobjdet, objdet_possible_labels = label_image.init_tf2(args["obj_det_model"], args["numthreads"],
                                                            args["obj_det_labels"])
    interpreter, possible_labels = label_image.init_tf2(args["species_model"], args["numthreads"],
                                                        args["species_labels"])

    print('press esc to quit')

    # main loop ******
    while True:  # while escape key is not pressed
        try:
            motionb, img, gray, graymotion, thresh = motion_detector.detect(cv2, cap, first_img, args["minarea"])
        except:
            logging.critical('camera failed on motion capture and compare of second image, ending program')
            return

        if motionb:  # motion detected.
            det_confidences, det_labels, det_rects = label_image.object_detection(args["confidence"], img,
                                                                                  objdet_possible_labels, tfobjdet,
                                                                                  args["inputmean"], args["inputstd"])
            birdb = False
            tweetb = False
            combined_label = ''
            for i, det_confidence in enumerate(det_confidences):
                logtime = datetime.now().strftime('%H:%M:%S')
                loginfo = " saw  {}: {:.2f}%".format(det_labels[i], det_confidence * 100)
                logging.info(logtime + loginfo)
                print(logtime, loginfo)
                if det_labels[i] == "bird" and (det_confidence >= args["confidence"] or birdb):
                    birdb = True
                    (startX, startY, endX, endY) = label_image.scale_rect(img, det_rects[i])  # x,y coord bounding box
                    logging.info(str(startX) + ' ' + str(startY) + ' ' +
                                 str(endX) + ' ' + str(endY))  # log rect size, use for bird size
                    logging.info(str( ((startX-startY) * (endX - endY)) ))
                    size, perarea = birdsize(args, startX, startY, endX, endY)
                    print (size, perarea)
                    # ts_img = img[startY:endY, startX:endX]  # extract image of bird
                    ts_img = img  # try maintaining size of org image for better recognition
                    tfconfidence, birdclass = label_image.set_label(ts_img, possible_labels, interpreter,
                                                                    args["inputmean"], args["inputstd"])

                    # draw bounding boxes and display label if it is a bird
                    if tfconfidence >= args["bconfidence"]:  # high confidence in species
                        tweetb = True
                        label = "{}: {:.2f}% / {:.2f}%".format(birdclass, tfconfidence * 100, det_confidence * 100)
                    else:
                        loginfo = "----species {}: {:.2f}% / {:.2f}%".format(birdclass,
                                                                        tfconfidence * 100, det_confidence * 100)
                        logging.info(loginfo)
                        print(loginfo)
                        label = "{}: {:.2f}%".format("bird", det_confidence * 100)
                        birdclass = 'bird'

                    combined_label = combined_label + ' ' + label  # build label for multi birds in one photo
                    cv2.rectangle(img, (startX, startY), (endX, endY), colors[i], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15  # adjust label loc if too low
                    cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

                    # keep track of what we've seen in the last 10 minutes
                    if birdclass in birds_found:  # seen it
                        logdate = starttime.strftime('%H:%M:%S')
                        loginfo = label + ' last seen at: '
                        logging.info(loginfo + logdate)
                        if (datetime.now().timestamp() - starttime.timestamp()) >= 600:  # 10 min elapsed in seconds;
                            starttime = datetime.now()  # reset timer
                            birds_found = []  # clear birds found
                    else:  # something new is at the feeder
                        birds_found.append(birdclass)

            if birdb:  # if object detection saw a bird draw the results
                cv2.imshow('obj detection', img)  # show all birds in pic with labels

            # image contained a bird and species label, tweet it
            if tweetb and (datetime.now().timestamp() - starttime.timestamp() >= 600):  # wait 10 min in seconds
                logdate = starttime.strftime('%H:%M:%S')
                logging.info(logdate + '**** tweeted ' + label)
                print(logdate, '**** tweeted', label)
                cv2.imshow('tweeted', img)  # show all birds in pic with labels
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


def birdsize(args, startX, startY, endX, endY):
    birdarea = abs((startX - startY) * (endX - endY))
    scrarea = args['screenheight'] * args['screenwidth']
    perarea = (birdarea / scrarea) * 100
    if perarea >= 20:  # large bird
        size = 'L'
    elif perarea >= 15:  # medium bird
        size = 'M'
    else:  # small bird
        size = 'S'

    logging.info(str(birdarea) + ' ' + str(scrarea) + ' ' + str(perarea) + ' ' + size)
    return size, perarea


def birdcolor(img, startX, startY, endX, endY):
    # from pyimagesearch.com color detection
    # define the list of boundaries.  Set sets of Green Blue Red GBR defining lower and upper bounds
    i = 0
    colorcount = {}
    boundaries = [
        ([17, 15, 100], [50, 56, 200]),  # Red
        ([86, 31, 4], [220, 88, 50]),  # Blue
        ([25, 146, 190], [62, 174, 250]),  # Yellow
        ([103, 86, 65], [145, 133, 128])  # Gray
    ]

    # loop over the boundaries
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")  # create NumPy arrays from the boundaries
        upper = np.array(upper, dtype="uint8")
        # find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(img, lower, upper)
        maskimg = cv2.bitwise_and(img, img, mask=mask)
        colorcount[i] = maskimg.any(axis=-1).countnonzero()  # count non-black pixels in image
        i += 1

    cindex = np.where(colorcount == np.amax(colorcount))
    return cindex


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # video setup
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--minarea", type=int, default=2, help="motion threshold, lower more")
    ap.add_argument("-sw", "--screenwidth", type=int, default=320, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=240, help="max screen height")

    # object detection model setup
    ap.add_argument('-om', "--obj_det_model",
                    default='/home/pi/birdclass/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite')
    ap.add_argument('-p', '--obj_det_labels',
                    default='/home/pi/PycharmProjects/pyface2/lite-model_ssd_mobilenet_v1_1_metadata_2_labelmap.txt')

    # species model setup
    ap.add_argument('-m', '--species_model',
                    default='/home/pi/PycharmProjects/pyface2/birdspecies-s-224-93.15.tflite',
                    help='.tflite model to be executed')
    ap.add_argument('-l', '--species_labels',
                    default='/home/pi/PycharmProjects/pyface2/birdspecies-13.txt',
                    help='name of file containing labels')

    # tensor flow input arguements
    ap.add_argument('--inputmean', default=127.5, type=float, help='Tensor input_mean')
    ap.add_argument('--inputstd', default=127.5, type=float, help='Tensor input standard deviation')
    ap.add_argument('--numthreads', default=None, type=int, help='Tensor number of threads')

    # confidence settings for object detection and species bconfidence
    ap.add_argument('-c', '--confidence', type=float, default=0.80)
    ap.add_argument('-bc', '--bconfidence', type=float, default=0.95)

    # set pan tilt control model
    ap.add_argument('--panb', default=False, type=bool, help='activate pan tilt mechanism')

    logging.basicConfig(filename='birdclass.log', format='%(asctime)s - %(message)s', level=logging.INFO)
    arguments = vars(ap.parse_args())
    print(datetime.now().strftime('%H:%M:%S'))
    bird_detector(arguments)
