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
import cv2
import PanTilt9685
import imutils  # CV helper functions
import argparse  # arguement parser
import tflite_runtime.interpreter as tflite
import numpy as np

# parse arguements
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# setup pan tilt and initialize variables
currpan, currtilt, pwm = PanTilt9685.init_pantilt()
SCR_W = 224
SCR_H = 224

cap = cv2.VideoCapture(0)  # capture
cap.set(3, SCR_W)  # set Width
cap.set(4, SCR_H)  # set Height
first_img = None  # init first img

# set up interpreters
# faceCascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
birdCascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades_birs/birds/xml')

# tensor flow lite setup
interpreter = tflite.interpreter(model_path='/home/pi/birdclass/lite-model_aiy_vision_classifier_birds_V1_3.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print('press esc to quit')

while True:  # while escape key is not pressed
    ret, img = cap.read()  # capture an image from the camera
    img = cv2.flip(img, -1)  # mirror image; comment out if not needed for your camera
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert image to gray scale BGR for OpenCv recognition
    graymotion = cv2.GaussianBlur(gray, (21, 21), 0)  # smooth out image for motion detection
    if first_img is None:
        first_img = graymotion

    # motion detection
    # compute the absolute difference between the current frame and
    # first frame
    imgdelta = cv2.absdiff(first_img, graymotion)
    thresh = cv2.threshold(imgdelta, 25, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue

        # motion detected
        # compute the bounding box for the contour, draw motion on frame
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # look for a face or other object if motion is detected
        # faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
        birds = birdCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

        # control pan title mechanism
        # currpan, currtilt = PanTilt9685.trackobject(pwm, cv2, currpan, currtilt, img, gray, faces, SCR_H, SCR_W)

    cv2.imshow('video', img)
    cv2.imshow('gray', graymotion)
    cv2.imshow('threshold', thresh)

    # check for esc key and quit if pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()



#Load input image
input_image = cv2.imread("/path/")
input_shape = input_details[0]['shape']
#Reshape input_image
np.reshape((input_image,input_shape)

#Set the value of Input tensor
interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()

#prediction for input data
output_data = interpreter.get_tensor(output_details[0]['index'])
fire_probability = output_data[0][0] * 100 #prediction probability
