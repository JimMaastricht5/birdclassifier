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
import cv2  # open cv 2
import label_image  # code to init tensor flow model and classify bird type
import PanTilt9685  # pan tilt control code
import motion_detector  # motion detector helper functions
import tweeter  # twitter helper functions
import argparse  # argument parser
import numpy as np
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

        if motionb:
            # look for objects if motion is detected
            det_confidences, det_labels, det_rects = label_image.object_detection(args["confidence"], img,
                                                                                  objdet_possible_labels, tfobjdet,
                                                                                  args["inputmean"], args["inputstd"])
            for i, det_confidence in enumerate(det_confidences):
                if det_confidence > args["confidence"]:
                    # then compute the (x, y)-coordinates of the bounding box
                    (startX, startY, endX, endY) = label_image.scale_rect(img, det_rects[i])

                    if det_labels[i] == "16.bird":
                        ts_img = img[startY:endY, startX:endX]  # extract image of bird
                        cv2.imwrite("tsimg.jpg", ts_img)  # write out files to disk for debugging and tensor feed
                        tfconfidence, birdclass = label_image.set_label(ts_img, possible_labels, interpreter,
                                                                        args["inputmean"], args["inputstd"])
                    else:  # not a bird
                        tfconfidence = det_confidence[i]
                        birdclass = det_labels[i]

                    # draw bounding boxes and display label
                    label = "{}: {:.2f}% ".format(birdclass, tfconfidence * 100)
                    cv2.rectangle(img, (startX, startY), (endX, endY), colors[i], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15  # adjust label loc if too low
                    cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
                    cv2.imwrite("img.jpg", img)

                    if det_labels[i] == "16.bird":  # share what you see
                        tw_img = open('cardinal.jpg', 'rb')
                        tweeter.post_image(twitter, label, tw_img)

            if args["panb"]:
                currpan, currtilt = PanTilt9685.trackobject(pwm, cv2, currpan, currtilt, img,
                                                            (startX, startY, endX, endY),
                                                            args["screenwidth"], args["screenheight"])

        cv2.imshow('video', img)
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
    ap.add_argument("-a", "--minarea", type=int, default=20, help="minimum area size")
    ap.add_argument("-sw", "--screenwidth", type=int, default=640, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=480, help="max screen height")
    ap.add_argument('-om', "--objmodel", default='/home/pi/birdclass/ssd_mobilenet_v1_1_metadata_1.tflite')
    ap.add_argument('-p', '--objlabels', default='/home/pi/birdclass/mscoco_label_map.txt')
    ap.add_argument('-c', '--confidence', type=float, default=0.5)
    ap.add_argument('-m', '--modelfile', default='/home/pi/birdclass/mobilenet_tweeters.tflite',
                    help='.tflite model to be executed')
    ap.add_argument('-l', '--labelfile', default='/home/pi/birdclass/class_labels.txt',
                    help='name of file containing labels')
    ap.add_argument('--inputmean', default=127.5, type=float, help='Tensor input_mean')
    ap.add_argument('--inputstd', default=127.5, type=float, help='Tensor input standard deviation')
    ap.add_argument('--numthreads', default=None, type=int, help='Tensor number of threads')
    ap.add_argument('--panb', default=False, type=bool, help='activate pan tilt mechanism')
    ap.add_argument('-i', '--image', default='/home/pi/birdclass/cardinal.jpg',
                                            help='image to be classified')

    arguments = vars(ap.parse_args())

    bird_detector(arguments)
