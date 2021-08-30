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
import math
from PIL import Image

import image_proc

try:
    import picamera
    import picamera.array
except:
    print('picamera import fails on windows')
    pass


# create in-memory stream, close stream when operation is complete
def capture_image(camera):
    stream = io.BytesIO()
    camera.capture(stream, 'jpeg')
    stream.seek(0)
    img = Image.open(stream)
    # img.save('testcap_motion.jpg')
    return img


# Create the camera object
# capture first image and gray scale/blur for baseline motion detection
def init(args):
    camera = picamera.PiCamera()
    if args.screenwidth != 0:  # use specified height and width or default values if not passed
        camera.resolution = (args.screenheight, args.screenwidth)
    camera.vflip = args.flipcamera
    camera.framerate = args.framerate
    # to capture consistent images, wait and fix values
    time.sleep(2)  # Wait for the automatic gain control to settle
    # camera.shutter_speed = camera.exposure_speed
    # camera.exposure_mode = 'off'
    # g = camera.awb_gains
    # camera.awb_mode = 'off'
    # camera.awb_gains = g
    img = capture_image(camera)  # capture img of type PIL
    gray = image_proc.grayscale(img)  # convert image to gray scale for motion detection
    graymotion = image_proc.gaussianblur(gray)  # smooth out image for motion detection
    return camera, graymotion


# once first image is captured call motion detector in a loop to find each subsequent image
# motion detection, compute the absolute difference between the current frame and first frame
def detect(camera, first_img, min_area):
    img = capture_image(camera)
    grayimg = image_proc.grayscale(img)  # convert image to gray scale
    grayblur = image_proc.gaussianblur(grayimg)  # smooth out image for motion detection
    imgdelta = image_proc.compare_images(first_img, grayblur)
    # print(f'image entropy is{image_entropy(imgdelta)}')
    return (image_entropy(imgdelta) >= min_area), img


# determine change between static image and new frame
def image_entropy(image):
    histogram = image.histogram()
    histlength = sum(histogram)
    probability = [float(h) / histlength for h in histogram]
    return -sum([p * math.log(p, 2) for p in probability if p != 0])
