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


class MotionDetector:
    def __init__(self, args, save_test_img=False):
        self.min_area = args.minarea
        self.camera = picamera.PiCamera()
        if args.screenwidth != 0:  # use specified height and width or default values if not passed
            self.camera.resolution = (args.screenheight, args.screenwidth)
        self.camera.vflip = args.flipcamera
        self.camera.framerate = args.framerate
        time.sleep(2)  # Wait for the automatic gain control to settle
        self.shutterspeed = self.camera.exposure_speed

        self.stream = io.BytesIO()
        self.img = self.capture_image()  # capture img of type PIL
        self.first_img = self.img.copy()
        self.gray = image_proc.grayscale(self.img)  # convert image to gray scale for motion detection
        self.graymotion = image_proc.gaussianblur(self.gray)  # smooth out image for motion detection
        if save_test_img:
            self.img.save('testcap_motion.jpg')

    # create in-memory stream, close stream when operation is complete
    def capture_image(self):
        self.camera.capture(self.stream, 'jpeg')
        self.stream.seek(0)
        img = Image.open(self.stream)
        return img

    # once first image is captured call motion detector in a loop to find each subsequent image
    # motion detection, compute the absolute difference between the current frame and first frame
    def detect(self):
        img = self.capture_image()
        grayimg = image_proc.grayscale(img)  # convert image to gray scale
        grayblur = image_proc.gaussianblur(grayimg)  # smooth out image for motion detection
        imgdelta = image_proc.compare_images(self.first_img, grayblur)
        return (self.image_entropy(imgdelta) >= self.min_area), img

    def stop(self):
        self.camera.close()
        return

    # determine change between static image and new frame
    def image_entropy(self, image_delta):
        histogram = image_delta.histogram()
        histlength = sum(histogram)
        probability = [float(h) / histlength for h in histogram]
        return -sum([p * math.log(p, 2) for p in probability if p != 0])

    def capture_stream(self, frame_rate=30, stream_frames=200, filename='birds.gif'):
        """
        function returns a list of images

        :param frame_rate: int value with number of images per second from camera setting
        :param stream_frames: int value with number of frames to capture
        :param filename: name of file to save gif under
        :return frames: images is a list containing a number of PIL jpg image
        :return gif: animated gif of images in images list
        """
        frames = []
        for image_num in (0, stream_frames):
            self.camera.capture(self.stream, 'jpeg')
            self.stream.seek(0)
            frames.append(Image.open(self.stream))
            frame_one = frames[0]
            ml_sec = 1000000 * stream_frames / frame_rate  # frames / rate, 200 /30 = 5 sec * 1,000,000 = ml sec
            frame_one.save(filename, format="GIF", append_images=frames,
                           save_all=True, duration=ml_sec, loop=0)  # loop=0 replays gif over and over
            gif = open(filename, 'rb')  # reload gif
        return frames, gif


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # camera settings
    ap.add_argument("-fc", "--flipcamera", type=bool, default=False, help="flip camera image")
    ap.add_argument("-sw", "--screenwidth", type=int, default=640, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=480, help="max screen height")
    ap.add_argument("-fr", "--framerate", type=int, default=30, help="frame rate for camera")
    arguments = ap.parse_args()

    motion_detector = MotionDetector(args=arguments)
    frames_test, gif_test = motion_detector.capture_stream()

    # Create the camera object
    # capture first image and gray scale/blur for baseline motion detection
    # def init(args):
    #     camera = picamera.PiCamera()
    #     if args.screenwidth != 0:  # use specified height and width or default values if not passed
    #         camera.resolution = (args.screenheight, args.screenwidth)
    #     camera.vflip = args.flipcamera
    #     camera.framerate = args.framerate
    #     time.sleep(2)  # Wait for the automatic gain control to settle
    #     print(f'shutter speed is {camera.exposure_speed}')
    #     img = capture_image(camera)  # capture img of type PIL
    #     print('camera initialized and gray image created... ')
    #     # img.save('testcap_motion.jpg')
    #     gray = image_proc.grayscale(img)  # convert image to gray scale for motion detection
    #     graymotion = image_proc.gaussianblur(gray)  # smooth out image for motion detection
    #     return camera, graymotion
