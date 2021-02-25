# MIT License
#
# 2021 Jim Maastricht
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# code to handle motion detection for pyface and tweetercam
# pass in opencv, capture settings, and first_img
# first_img should be without motion.
# compare gray scale image to first image.  If different than motion
# return image captured, gray scale, guassian blured version, thresholds, first_img, and countours
# code by JimMaastricht5@gmail.com based on https://www.pyimagesearch.com/category/object-tracking/
# import cv2
import imutils


# capture first image and gray scale/blur for baseline motion detection
def init(flipb, cv2, cap):
    ret, img = cap.read()  # capture an image from the camera
    if flipb:
        img = cv2.flip(img, -1)  # mirror image; comment out if not needed for your camera
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert image to gray scale BGR for OpenCv recognition
    graymotion = cv2.GaussianBlur(gray, (21, 21), 0)  # smooth out image for motion detection
    return graymotion


# once first image is captured call motion detector in a loop to find each subsequent image
# compare image to first img; if different than motion
def detect(flipb, cv2, cap, first_img, min_area):
    motionb = False
    ret, img = cap.read()  # capture an image from the camera
    if flipb:
        img = cv2.flip(img, -1)  # mirror image; comment out if not needed for your camera
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert image to gray scale BGR for OpenCv recognition
    grayblur = cv2.GaussianBlur(grayimg, (21, 21), 0)  # smooth out image for motion detection

    # motion detection, compute the absolute difference between the current frame and first frame
    imgdelta = cv2.absdiff(first_img, grayblur)
    threshimg = cv2.threshold(imgdelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours on the image
    threshimg = cv2.dilate(threshimg, None, iterations=2)
    cnts = cv2.findContours(threshimg.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = imutils.grab_contours(cnts)  # set of contours showing motion
    for c in cnts:  # loop over countours if too small ignore it
        # print(cv2.contourArea(c))
        if cv2.contourArea(c) >= min_area:
            motionb = True

    # return motionb, img, grayimg, grayblur, threshimg
    return motionb, img
