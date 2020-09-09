# Haar Cascade Face detection with OpenCV
# based on https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
# cascades at: https://github.com/Itseez/opencv/tree/master/data/haarcascades
# Adpted by Jim Maastricht jimmaastricht5@gmail.com to incorporate pantilt functionality and motion detection

# special notes when setting up the code on a rasberry pi 4 9/9/20
# encountered an "undefined symbol: __atomic_fetch_add8" error related to libatomic.so on OpenCv version w/Raspberry Pi
# install OpenCv version 4.1.0.25 to resolve the issue on the pi
# packages: pan tilt uses PCA9685-Driver
import cv2  # opencv2
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


def face_detector(args):
    # setup pan tilt and initialize variables
    currpan, currtilt, pwm = PanTilt9685.init_pantilt()

    facecascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)  # capture video image
    cap.set(3, args["screen-width"])  # set screen width
    cap.set(4, args["screen-height"])  # set screen height

    twitter = tweeter.init(api_key, api_secret_key, access_token, access_token_secret)
    first_img = motion_detector.init(cv2, cap)

    print('press esc to quit')
    while True:  # while escape key is not pressed
        motionb, ret, img, gray, graymotion, thresh, first_img = \
            motion_detector.detect(cv2, cap, first_img, args["min-area"])
        
        if motionb:  # motion detected boolean = True
            # look for a face or other object if motion is detected
            # higher scale is faster, higher min n more accurate but more false neg
            faces = facecascade.detectMultiScale(gray, scaleFactor=1.0485258, minNeighbors=5)
            for (x, y, w, h) in faces:
                rect = (x, y, (x + w), (y + h))
                cv2.rectangle(img, rect, (0, 255, 0), 2)
                face_img = img[int(x):int(y), int(x + w):int(y + h)]
                # roi_gray = gray[y:y + h, x:x + w]
                # roi_color = img[y:y + h, x:x + w]
                twitter.post_image("found face", face_img)

            currpan, currtilt = PanTilt9685.trackobject(pwm, cv2, currpan, currtilt, img, faces,
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
    ap.add_argument("-sw", "--screen-width", type=int, default=640, help="max screen width")
    ap.add_argument("-sh", "--screen-height", type=int, default=380, help="max screen height")
    arguments = vars(ap.parse_args())

    face_detector(arguments)
