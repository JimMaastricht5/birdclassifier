# Haar Cascade Face detection with OpenCV
#    Based on tutorial by pythonprogramming.net
#    Visit original post: https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
# Adapted by Marcelo Rovai - MJRoBot.org @ 7Feb2018
# Adpted by Jim Maastricht to incorporate pantilt functionality
# cascades at: https://github.com/Itseez/opencv/tree/master/data/haarcascades
# Q: What do I do when I encounter an "undefined symbol: __atomic_fetch_add8" error related to libatomic.so?
# A: The pip install has been giving readers troubles since OpenCV 4.1.1
# (around the November 2019 timeframe). Be sure to install version 4.1.0.25 :
# packages: PCA9685-Driver, opencv-python=4.1.0.25
import cv2  # opencv2
import PanTilt9685  # pan tilt control code
import motion_detector  # motion detector helper functions
import tweeter  # twitter helper functions
import imutils  # CV helper functions
import argparse  # arguement parser
# from twython import Twython; needed?
from auth import (
    api_key,
    api_secret_key,
    access_token,
    access_token_secret
)

def face_detector(args):
    # setup pan tilt and initialize variables
    currpan, currtilt, pwm = PanTilt9685.init_pantilt()
    SCR_W = 640
    SCR_H = 480

    faceCascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)  # capture video image
    cap.set(3, SCR_W)  # set Width
    cap.set(4, SCR_H)  # set Height

    twitter = tweeter.init(api_key, api_secret_key, access_token, access_token_secret)
    first_img = motion_detector.init(cv2, cap)

    print('press esc to quit')
    while True:  # while escape key is not pressed
        ret, img, gray, graymotion, thresh, first_img, cnts = motion_detector.detect(cv2, cap, first_img)

        cnts = imutils.grab_contours(cnts)  # set of contours showing motion
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                continue

            # motion detected
            # compute the bounding box for the contour, draw motion on frame
            # (x, y, w, h) = cv2.boundingRect(c)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # look for a face or other object if motion is detected
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
            for (x, y, w, h) in faces:
                rect = (x, y, (x + w), (y + h))
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(img, rect, (0, 255, 0), 2)
                face_img = img[int(x):int(x + w), int(y):int(y + w)]  # one of these two shoudl work?
                face_img = img(rect)  # one of these two should work?
                twitter.post_image("found face", face_img)

            # control pan title mechanism
            currpan, currtilt = PanTilt9685.trackobject(pwm, cv2, currpan, currtilt, img, gray, faces, SCR_H, SCR_W)

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
    # parse arguements
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    args = vars(ap.parse_args())

    face_detector(args)
