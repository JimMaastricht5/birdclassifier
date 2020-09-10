# motion detector with builtin bird detection and bird classification
# built by JimMaastricht5@gmail.com
# uses Haar Cascade for bird detection and tensorflow lite model for bird classification
# tensor flow model built using tools from tensor in colab notebook birdclassifier.ipynb
# model trained to find birds common to feeders in WI using caltech bird dataset 2011
# 017.Cardinal
# 029.American_Crow
# 047.American_Goldfinch
# 049.Boat_tailed_Grackle
# 073.Blue_Jay
# 076.Dark_eyed_Junco
# 094.White_breasted_Nuthatch
# 118.House_Sparrow
# 129.Song_Sparrow
# 191.Red_headed_Woodpecker
# 192.Downy_Woodpecker

# special notes when setting up the code on a rasberry pi 4 9/9/20
# encountered an "undefined symbol: __atomic_fetch_add8" error related to libatomic.so on OpenCv version w/Raspberry Pi
# install OpenCv version 4.1.0.25 to resolve the issue on the pi
# packages: pan tilt uses PCA9685-Driver
import cv2  # opencv2
import time
import numpy as np
from PIL import Image
import tensorflow as tf  # TF2
import PanTilt9685  # pan tilt control code
import motion_detector  # motion detector helper functions
import tweeter  # twitter helper functions
import argparse  # argument parser
from auth import (
    api_key,
    api_secret_key,
    access_token,
    access_token_secret
)  # import twitter keys


def bird_detector(args):
    # setup pan tilt and initialize variables
    currpan, currtilt, pwm = PanTilt9685.init_pantilt()

    birdCascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades_birs/birds/xml')
    cap = cv2.VideoCapture(0)  # capture video image
    cap.set(3, args["screen-width"])  # set screen width
    cap.set(4, args["screen-height"])  # set screen height

    twitter = tweeter.init(api_key, api_secret_key, access_token, access_token_secret)
    first_img = motion_detector.init(cv2, cap)

    # tensor flow lite setup
    interpreter = tflite.interpreter(model_path='/home/pi/birdclass/lite-model_aiy_vision_classifier_birds_V1_3.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print('press esc to quit')
    while True:  # while escape key is not pressed
        motionb, ret, img, gray, graymotion, thresh, first_img = \
            motion_detector.detect(cv2, cap, first_img, args["min-area"])

        if motionb:  # motion detected boolean = True
            # look for object if motion is detected
            # higher scale is faster, higher min n more accurate but more false neg 3-5 reasonable range
            birds = birdCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
            for (x, y, w, h) in birds:
                rect = (x, y, (x + w), (y + h))
                cv2.rectangle(img, rect, (0, 255, 0), 2)
                # bird_img = img[x):(y), (x + w):(y + h)]  # old
                bird_img = img[y:y + h, x:x + w] # try this?
                twitter.post_image("found bird", bird_img)

            currpan, currtilt = PanTilt9685.trackobject(pwm, cv2, currpan, currtilt, img, birds,
                                                        args["screen-width"], args["screen-height"])

        cv2.imshow('video', img)
        cv2.imshow('gray', graymotion)
        cv2.imshow('threshold', thresh)

        # check for esc key and quit if pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    ap.add_argument("-sw", "--screen-width", type=int, default=300, help="max screen width")
    ap.add_argument("-sh", "--screen-height", type=int, default=300, help="max screen height")
    arguments = vars(ap.parse_args())

    bird_detector(arguments)
