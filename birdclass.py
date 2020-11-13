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
(kaggle)
# 300.American GoldFinch
# 301.Baltimore Oriole
# 302.Black Capped Chickadee
# 303.Brown Thrasher
# 304.Chipping Sparrow
# 305.Common Starling
# 306.Dark Eyed Junko
# 307.Eastern Bluebird
# 308.Eastern Meadowlark
# 309.Eastern Towee
# 310.Evening Grosbeak
# 311.Green Jay
# 312.House Finch
# 313.Northern Cardinal
# 314.Purple Finch
# 315.Robin
# 316.Tit Mouse


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
from auth import (
    api_key,
    api_secret_key,
    access_token,
    access_token_secret
)  # import twitter keys


def bird_detector(args):
    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    colors = np.random.uniform(0, 255, size=(11, 3))  # random colors for bounding boxes
    birds_found = []
    starttime = time.time()
    motioncheck = time.time()

    # setup pan tilt and initialize variables
    if args["panb"]:
        currpan, currtilt, pwm = PanTilt9685.init_pantilt()

    # initial video capture, screen size, and grab first image (no motion)
    if args["image"] == "":
        cap = cv2.VideoCapture(0)  # capture video image
        cap.set(3, args["screenwidth"])  # set screen width
        cap.set(4, args["screenheight"])  # set screen height
        first_img = motion_detector.init(cv2, cap)
    else:
        first_img = cv2.imread(args["image"])  # testing code

    twitter = tweeter.init(api_key, api_secret_key, access_token, access_token_secret)  # init twitter api

    # tensor flow lite setup; TF used to classify detected birds; need to convert that to tf lite
    interpreter, possible_labels = label_image.init_tf2(args["modelfile"], args["numthreads"], args["labelfile"])
    tfobjdet, objdet_possible_labels = label_image.init_tf2(args["objmodel"], args["numthreads"], args["objlabels"])

    print('press esc to quit')
    while True:  # while escape key is not pressed
        if args["image"] == "":
            motionb, img, gray, graymotion, thresh = motion_detector.detect(cv2, cap, first_img, args["minarea"])
        else:  # testing code
            motionb = True
            first_img = cv2.imread(args["image"])
            img = cv2.imread(args["image"])

        if ((time.time() - motioncheck) > 1):  # motion detection not working w/fisheye lens
        # if motionb:  # and ((time.time() - motioncheck) > 1):
            motioncheck = time.time()
            # look for objects if motion is detected
            det_confidences, det_labels, det_rects = label_image.object_detection(args["confidence"], img,
                                                                                  objdet_possible_labels, tfobjdet,
                                                                                  args["inputmean"], args["inputstd"])
            for i, det_confidence in enumerate(det_confidences):
                print(det_confidence, det_labels[i])
                if det_confidence > args["confidence"]:
                    # then compute the (x, y)-coordinates of the bounding box
                    (startX, startY, endX, endY) = label_image.scale_rect(img, det_rects[i])

                    if det_labels[i] == "bird":
                        ts_img = img[startY:endY, startX:endX]  # extract image of bird
                        # cv2.imwrite("tsimg.jpg", ts_img)  # write out files to disk for debugging and tensor feed
                        tfconfidence, birdclass = label_image.set_label(ts_img, possible_labels, interpreter,
                                                                        args["inputmean"], args["inputstd"])
                    else:  # not a bird
                        tfconfidence = det_confidence
                        birdclass = det_labels[i]

                    # draw bounding boxes and display label
                    if tfconfidence >= args["confidence"]: # high confidence in species
                        label = "{}: {:2f}% ".format(birdclass, tfconfidence * 100)
                    else:
                        label = "{}: {:2f}% ".format("bird", det_confidence * 100)

                    cv2.rectangle(img, (startX, startY), (endX, endY), colors[i], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15  # adjust label loc if too low
                    cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
                    cv2.imshow('obj detection', img)

                    if det_labels[i] == "bird":  # share what you see
                        if birdclass in birds_found:  # seen it
                            elapsed_time = time.time()
                            if elapsed_time - starttime >= 300:  # elapsed time in seconds; give it 5 minutes
                                starttime = time.time()  # rest timer
                                birds_found = []  # clear birds found
                        else:
                            cv2.imwrite("img.jpg", img)  # write out image for debugging and testing
                            tw_img = open('img.jpg', 'rb')
                            tweeter.post_image(twitter, label, tw_img)
                            birds_found.append(birdclass)

            if args["panb"]:
                currpan, currtilt = PanTilt9685.trackobject(pwm, cv2, currpan, currtilt, img,
                                                            (startX, startY, endX, endY),
                                                            args["screenwidth"], args["screenheight"])
        ret, videoimg = cap.read()
        videoimg = cv2.flip(videoimg, -1)  # mirror image; comment out if not needed for your camera
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
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--minarea", type=int, default=5, help="minimum area size")
    ap.add_argument("-sw", "--screenwidth", type=int, default=320, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=240, help="max screen height")
    ap.add_argument('-om', "--objmodel", default='/home/pi/birdclass/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite')
    ap.add_argument('-p', '--objlabels',
                    default='/home/pi/birdclass/lite-model_ssd_mobilenet_v1_1_metadata_2_labelmap.txt')
    ap.add_argument('-c', '--confidence', type=float, default=0.50)
    ap.add_argument('-m', '--modelfile', default='/home/pi/birdclass/mobilenet_tweeters.tflite',
                    help='.tflite model to be executed')
    ap.add_argument('-l', '--labelfile', default='/home/pi/birdclass/class_labels.txt',
                    help='name of file containing labels')
    ap.add_argument('--inputmean', default=127.5, type=float, help='Tensor input_mean')
    ap.add_argument('--inputstd', default=127.5, type=float, help='Tensor input standard deviation')
    ap.add_argument('--numthreads', default=None, type=int, help='Tensor number of threads')
    ap.add_argument('--panb', default=False, type=bool, help='activate pan tilt mechanism')
    # ap.add_argument('-i', '--image', default='/home/pi/birdclass/cardinal.jpg',
    #                                         help='image to be classified')
    ap.add_argument('-i', '--image', default='', help='image to be classified')

    arguments = vars(ap.parse_args())

    bird_detector(arguments)
