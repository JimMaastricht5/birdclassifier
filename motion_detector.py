# MIT License
#
# 2024 Jim Maastricht
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
import numpy as np
import image_proc

try:
    import picamera
    # import picamera.array
except Exception as e:
    print(e)
    print('continuing motion detection setup....')
    pass


class MotionDetector:
    def __init__(self, motion_min_area=4, screenwidth=640, screenheight=480, flip_camera=False,
                 iso=800, first_img_name='capture.jpg'):

        print('initializing camera')
        self.camera = picamera.PiCamera()
        self.min_area = motion_min_area
        if screenwidth != 0:  # use specified height and width or default values if not passed
            self.camera.resolution = (screenheight, screenwidth)
        self.camera.vflip = flip_camera
        self.camera.iso = iso  # iso 800 for less blur
        time.sleep(2)  # Wait for the automatic gain control to settle
        # self.shutterspeed = self.camera.exposure_speed
        # self.camera.exposure_mode = 'off'
        # self.camera.exposure_mode = 'off'
        # self.gain = camera.awb_gains
        # self.camera.awb_mode = 'off'
        # self.camera.awb_gains = self.gain

        # set up first image. base for motion detection
        print(f'capturing first image: {first_img_name}')
        self.img_filename = first_img_name
        self.capture_image_with_file(filename=self.img_filename)  # capture img
        self.img = Image.open(self.img_filename)
        self.gray = image_proc.grayscale(self.img)  # convert image to gray scale for motion detection
        self.graymotion = image_proc.gaussianblur(self.gray)  # smooth out image for motion detection
        self.first_img = self.graymotion.copy()

        self.motion = False  # init motion detection boolean
        self.FPS = 0  # calculated frames per second
        print('camera setup completed')

    def capture_image_with_file(self, img_type='jpeg', filename='/home/pi/capture_image.jpg'):
        stream = io.BytesIO()
        self.camera.capture(stream, img_type, use_video_port=True)
        stream.seek(0)
        img = Image.open(stream)
        img.save(filename)
        return

    # grab and image and store in mem
    def capture_image_stream(self, img_type='jpeg'):
        stream = io.BytesIO()
        self.camera.capture(stream, img_type, use_video_port=True)
        stream.seek(0)
        img = Image.open(stream)
        return img

    def capture_stream(self, stream_frames=12):
        """
        function returns a list of images

        :param stream_frames: int value with number of frames to capture
        :return frames: images is a list containing a number of PIL jpg image
        """
        frames = []
        start_time = time.time()
        for image_num in range(stream_frames):
            img = self.capture_image_stream()
            frames.append(img)
        self.FPS = stream_frames / float(time.time() - start_time)
        return frames

    # grab an image using NP array: doesn't work!!!!
    def capture_image_np(self, img_type='jpeg'):
        height, width = self.camera.resolution
        img = np.empty((height, width, 3), dtype=np.uint8)
        # print(height, width)
        self.camera.capture(img, img_type)
        img_pil = image_proc.convert(img=img, convert_to='PIL')
        img_pil.save('/home/pi/birdclass/alt_camera_img.jpg')
        return img_pil

    # once first image is captured call motion detector in a loop to find each subsequent image
    # motion detection, compute the absolute difference between the current frame and first frame
    # if the difference is more than the tolerance we have something new in the frame aka motion
    def detect(self):
        try:  # trap any camera or image errors gracefully
            self.img = self.capture_image_stream()
            grayimg = image_proc.grayscale(self.img)  # convert image to gray scale
            grayblur = image_proc.gaussianblur(grayimg)  # smooth out image for motion detection
            imgdelta = image_proc.compare_images(self.first_img, grayblur)
            self.motion = (self.image_entropy(imgdelta) >= self.min_area)
        except Exception as e:
            self.motion = False
            print(e)
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


# if __name__ == '__main__':
#     # construct the argument parser and parse the arguments
#     ap = argparse.ArgumentParser()
#     # camera settings
#     ap.add_argument("-fc", "--flipcamera", type=bool, default=False, help="flip camera image")
#     ap.add_argument("-sw", "--screenwidth", type=int, default=640, help="max screen width")
#     ap.add_argument("-sh", "--screenheight", type=int, default=480, help="max screen height")
#     arguments = ap.parse_args()
#
#     motion_detector = MotionDetector(args=arguments)
#     frames_test = motion_detector.capture_stream()
