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
import io
import time
from PIL import Image

import image_proc

try:
    import picamera
    import picamera.array
except:
    print('picamera import fails on windows')
    pass

# create in-memory stream, close stream when operation is complete
def capture_image(flipb, camera):
    # with picamera.array.PiRGBArray(camera) as stream:
    stream = io.BytesIO()
    camera.capture(stream, 'jpeg')
    stream.seek(0)
    img = Image.open(stream)
    # stream.truncate()
    # img = image_proc.convert(img, "PIL")
    if flipb:
        img = image_proc.flip(img)
    return img

# Create the camera object
# capture first image and gray scale/blur for baseline motion detection
def init(flipb):
    camera = picamera.PiCamera()
    camera.resolution = (640, 480)
    # camera.start_preview()
    time.sleep(2)
    img = capture_image(flipb, camera)
    # img = image_proc.convert(img, "PIL")
    gray = image_proc.grayscale(img)  # convert image to gray scale for motion detection
    graymotion = image_proc.gaussianblur(gray)  # smooth out image for motion detection
    return camera, graymotion


# once first image is captured call motion detector in a loop to find each subsequent image
# compare image to first img; if different than motion
def detect(flipb, camera, first_img, min_area):
    motionb = False
    img = capture_image(flipb, camera)
    grayimg = image_proc.grayscale(img)  # convert image to gray scale
    grayblur = image_proc.gaussianblur(img)  # smooth out image for motion detection

    # motion detection, compute the absolute difference between the current frame and first frame
    # imgdelta = image_proc.compare_images(first_img, grayblur)
    # threshimg = cv2.threshold(imgdelta, 25, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours on the image
    # threshimg = cv2.dilate(threshimg, None, iterations=2)
    # cnts = cv2.findContours(threshimg.copy(), cv2.RETR_EXTERNAL,
    #                         cv2.CHAIN_APPROX_SIMPLE)
    
    # cnts = imutils.grab_contours(cnts)  # set of contours showing motion
    # for c in cnts:  # loop over countours if too small ignore it
    #     print(cv2.contourArea(c))
        # if cv2.contourArea(c) >= min_area:
        #     motionb = True

    # return motionb, img, grayimg, grayblur, threshimg
    motionb = True
    return motionb, img
