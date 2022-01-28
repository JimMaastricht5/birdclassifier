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
import copy
import numpy as np
import image_proc
import argparse

try:
    import picamera
    # import picamera.array
except Exception as e:
    print(e)
    print('continuing motion detection setup....')
    pass


class MotionDetector:
    def __init__(self, args, save_img=False):
        print('initializing camera')
        self.camera = picamera.PiCamera()
        self.stream = io.BytesIO()
        self.min_area = args.minarea
        if args.screenwidth != 0:  # use specified height and width or default values if not passed
            self.camera.resolution = (args.screenheight, args.screenwidth)
        self.camera.vflip = args.flipcamera
        # self.camera.framerate = args.framerate
        print('sleep to let camera settle')
        time.sleep(2)  # Wait for the automatic gain control to settle
        # self.shutterspeed = self.camera.exposure_speed
        print('capturing first image')
        self.img = self.capture_image()  # capture img
        self.gray = image_proc.grayscale(self.img)  # convert image to gray scale for motion detection
        self.graymotion = image_proc.gaussianblur(self.gray)  # smooth out image for motion detection
        self.first_img = self.graymotion.copy()
        self.motion = False
        self.save_img = save_img  # boolean to set physically save images to disk behavior
        if self.save_img:
            self.img.save('testcap_motion.jpg')
        print('camera setup completed')

    # org code
    def capture_image(self, img_type='jpg'):
        stream = io.BytesIO()
        self.camera.capture(stream, img_type)
        stream.seek(0)
        img = Image.open(stream)
        if img.size == 0:
            print('zero byte img!!!!')
        return img
    # going back to using a file...
    # def capture_image(self, img_type='jpeg'):
    #     self.camera.capture('cap.jpg', img_type)
    #     img = Image.open('cap.jpg')
    #     return img
    # # revised to carry stream as at class creation until end of process
    # def capture_image(self, img_type='jpeg'):
    #     self.camera.capture(self.stream, img_type)
    #     self.stream.seek(0)
    #     img = Image.open(self.stream)
    #     return img
    # grab an image from the open stream
    # def capture_image(self, img_type='jpeg'):
    #     # with io.BytesIO() as stream:
    #     stream = io.BytesIO
    #     self.camera.capture(stream, img_type)
    #     stream.seek(0)
    #     img = Image.open(stream)
    #     img_copy = img.copy()
    #     return img_copy

    # grab an image using NP array: doesn't work!!!!
    def capture_image_np(self, img_type='jpeg'):
        height, width = self.camera.resolution
        img = np.empty((height, width, 3), dtype=np.uint8)
        print(height, width)
        self.camera.capture(img, img_type)
        img_pil = image_proc.convert(img=img, convert_to='PIL')
        img_pil.save('alt_camera_img.jpg')
        return img_pil

    # once first image is captured call motion detector in a loop to find each subsequent image
    # motion detection, compute the absolute difference between the current frame and first frame
    # if the difference is more than the tolerance we have something new in the frame aka motion
    def detect(self):
        img = self.capture_image()
        grayimg = image_proc.grayscale(img)  # convert image to gray scale
        grayblur = image_proc.gaussianblur(grayimg)  # smooth out image for motion detection
        imgdelta = image_proc.compare_images(self.first_img, grayblur)
        self.img = img
        if self.save_img:
            self.img.save('capture.jpg')
        self.motion = (self.image_entropy(imgdelta) >= self.min_area)
        return self.motion

    def stop(self):
        self.camera.close()
        return

    # determine change between static image and new frame
    def image_entropy(self, image_delta):
        histogram = image_delta.histogram()
        histlength = sum(histogram)
        probability = [float(h) / histlength for h in histogram]
        return -sum([p * math.log(p, 2) for p in probability if p != 0])

    def capture_stream(self, stream_frames=15, save_img=False):
        """
        function returns a list of images

        :param stream_frames: int value with number of frames to capture
        :param save_img: bool True saves each image captured in the stream.  Slow!  default False
        :return frames: images is a list containing a number of PIL jpg image
        """
        frames = []
        for image_num in range(stream_frames):
            img = self.capture_image().copy()
            if save_img:
                img.save('/home/pi/birdclass/streamcap' + str(image_num) + '.jpg')
            frames.append(img)
        return frames


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # camera settings
    ap.add_argument("-fc", "--flipcamera", type=bool, default=False, help="flip camera image")
    ap.add_argument("-sw", "--screenwidth", type=int, default=640, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=480, help="max screen height")
    ap.add_argument("-fr", "--framerate", type=int, default=30, help="frame rate for camera")
    arguments = ap.parse_args()

    motion_detector = MotionDetector(args=arguments, save_img=True)
    frames_test = motion_detector.capture_stream(save_img=True)
