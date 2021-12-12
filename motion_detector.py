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
import argparse

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
    time.sleep(2)  # Wait for the automatic gain control to settle
    print(f'shutter speed is {camera.exposure_speed}')
    img = capture_image(camera)  # capture img of type PIL
    # img.save('testcap_motion.jpg')
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


def capture_stream(camera, frame_rate=30, stream_frames=30):
    """
    function returns a list of images

    :param camera: picamera object
    :param frame_rate: int value with number of images per second from camera setting
    :param stream_frames: int value with number of frames to capture
    :return frames: images is a list containing a number of PIL jpg image
    :return gif: animated gif of images in images list
    """
    frames = []
    stream = io.BytesIO()
    for image_num in (0, stream_frames):
        camera.capture(stream, 'jpeg')
        stream.seek(0)
        frames.append(Image.open(stream))
        frame_one = frames[0]
        ml_sec = 1/1000000 * stream_frames * frame_rate
        # duration in ml second for 1/30 of a second frame rate with 30 shots; should be 33,333?
        frame_one.save("birds.gif", format="GIF", append_images=frames,
                       save_all=True, duration=ml_sec, loop=0)  # duration in ml sec, loop zero loops the image forever
        gif = open('birds.gif', 'rb')  # reload gih
    return frames, gif


if __name__== '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # camera settings
    ap.add_argument("-fc", "--flipcamera", type=bool, default=False, help="flip camera image")
    ap.add_argument("-sw", "--screenwidth", type=int, default=640, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=480, help="max screen height")
    ap.add_argument("-fr", "--framerate", type=int, default=30, help="frame rate for camera")
    arguments = ap.parse_args()

    camera, graymotion = init(arguments)
    frames, gif = capture_stream(camera)
