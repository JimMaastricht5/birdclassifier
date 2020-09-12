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
import label_image  # code to init tensor flow model and classify bird type
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

    bird_cascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades_birds/birds1.xml')
    cap = cv2.VideoCapture(0)  # capture video image
    cap.set(3, args["screenwidth"])  # set screen width
    cap.set(4, args["screenheight"])  # set screen height

    twitter = tweeter.init(api_key, api_secret_key, access_token, access_token_secret)
    first_img = motion_detector.init(cv2, cap)

    # tensor flow lite setup
    interpreter, possible_labels = label_image.init_tf2(args["modelfile"], args["numthreads"], args["labelfile"])

    print('press esc to quit')
    while True:  # while escape key is not pressed
        motionb, img, gray, graymotion, thresh = \
            motion_detector.detect(cv2, cap, first_img, args["minarea"])

        if motionb:  # motion detected boolean = True
            # look for object if motion is detected
            # higher scale is faster, higher min n more accurate but more false neg 3-6 reasonable range
            birds = bird_cascade.detectMultiScale(gray, scaleFactor=1.0485258, minNeighbors=6)
            for (x, y, w, h) in birds:
                cv2.rectangle(img, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)
                bird_img = img[y:y + h, x:x + w]  # extract image of bird
                cv2.imwrite("temp.jpg", bird_img) # write out file to disk for debugging and tensor feed
                # run tensor flow lite model to id bird type
                ts_img = Image.open("temp.jpg") # reload from image; ensures matching disk to tensor
                confidence, label = label_image.set_label(ts_img, possible_labels, interpreter,
                                                          args["inputmean"], args["inputstd"])
                print(confidence, label)
                # twitter.post_image(confidence + " " + label, bird_img)

            currpan, currtilt = PanTilt9685.trackobject(pwm, cv2, currpan, currtilt, img, birds,
                                                        args["screenwidth"], args["screenheight"])

        cv2.imshow('video', img)
        # cv2.imshow('gray', graymotion)
        # cv2.imshow('threshold', thresh)
        # cv2.imshow('bird', bird_img)

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
    ap.add_argument("-a", "--minarea", type=int, default=20, help="minimum area size")
    ap.add_argument("-sw", "--screenwidth", type=int, default=640, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=480, help="max screen height")
    ap.add_argument('-m', '--modelfile', default='/home/pi/birdclass/mobilenet_tweeters.tflite',
                    help='.tflite model to be executed')
    ap.add_argument('-l', '--labelfile', default='/home/pi/birdclass/class_labels.txt',
                    help='name of file containing labels')
    ap.add_argument('--inputmean', default=127.5, type=float, help='Tensor input_mean')
    ap.add_argument('--inputstd', default=127.5, type=float, help='Tensor input standard deviation')
    ap.add_argument('--numthreads', default=None, type=int, help='Tensor number of threads')
    # ap.add_argument('-i', '--image', default='/home/pi/birdclass/cardinal.jpg',
    #                                help='image to be classified')

    arguments = vars(ap.parse_args())

    bird_detector(arguments)
