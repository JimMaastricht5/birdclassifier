# Haar Cascade Face detection with OpenCV
# based on https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
# cascades at: https://github.com/Itseez/opencv/tree/master/data/haarcascades
# Adpted by Jim Maastricht jimmaastricht5@gmail.com to incorporate pantilt functionality and motion detection

# special notes when setting up the code on a rasberry pi 4 9/9/20
# encountered an "undefined symbol: __atomic_fetch_add8" error related to libatomic.so on OpenCv version w/Raspberry Pi
# install OpenCv version 4.1.0.25 to resolve the issue on the pi
# packages: pan tilt uses PCA9685-Driver
import cv2  # opencv2
from archive import PanTilt9685
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
    cap.set(3, args["screenwidth"])  # set screen width
    cap.set(4, args["screenheight"])  # set screen height

    twitter = tweeter.init(api_key, api_secret_key, access_token, access_token_secret)
    first_img = motion_detector.init(cv2, cap)

    print('press esc to quit')
    while True:  # while escape key is not pressed
        motionb, img, gray, graymotion, thresh = \
            motion_detector.detect(cv2, cap, first_img, args["minarea"])
        
        if motionb:  # motion detected boolean = True
            # look for a face or other object if motion is detected
            # higher scale is faster, higher min n more accurate but more false neg
            faces = facecascade.detectMultiScale(gray, scaleFactor=1.0485258, minNeighbors=6)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)
                face_img = img[y:y + h, x:x + w]  # extract face
                cv2.imshow('face', face_img)
                cv2.imshow('video', img)
                # twitter.post_image("found face", face_img)

            currpan, currtilt = PanTilt9685.trackobject(pwm, cv2, currpan, currtilt, img, faces,
                                                        args["screenwidth"], args["screenheight"])


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
    ap.add_argument("-a", "--minarea", type=int, default=500, help="minimum area size")
    ap.add_argument("-w", "--screenwidth", type=int, default=640, help="max screen width")
    ap.add_argument("-g", "--screenheight", type=int, default=480, help="max screen height")
    arguments = vars(ap.parse_args())

    face_detector(arguments)
