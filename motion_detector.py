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
# import io
# import numpy as np
import time
import math
from PIL import Image
import image_proc
from picamera2 import Picamera2, Preview
from libcamera import Transform


class MotionDetector:
    def __init__(self, motion_min_area=4, screenwidth=640, screenheight=480, flip_camera=False,
                 first_img_name='capture.jpg', file_dest='assets'):

        print('initializing camera')
        self.camera2 = Picamera2()
        self.min_area = motion_min_area
        self.screenwidth = screenwidth
        self.screenheight = screenheight
        self.config = self.camera2.create_preview_configuration(main={"size": (screenheight, screenwidth)},
                                                                transform=Transform(vflip=flip_camera))
        self.camera2.configure(self.config)
        self.camera2.start()
        time.sleep(2)  # Wait for the automatic gain control to settle

        # set up first image. base for motion detection
        print(f'capturing first image: {first_img_name}')
        self.first_img_filename = first_img_name
        self.file_dest = file_dest
        self.img = self.capture_image_with_file()  # capture img
        self.gray = image_proc.grayscale(self.img)  # convert image to gray scale for motion detection
        self.graymotion = image_proc.gaussianblur(self.gray)  # smooth out image for motion detection
        self.first_img = self.graymotion.copy()
        self.motion = False  # init motion detection boolean
        self.FPS = 0  # calculated frames per second
        print('camera setup completed')

    def capture_image_with_file(self, filename=None):
        filename = self.first_img_filename if filename is None else filename
        self.camera2.capture_file(f'{self.file_dest}/{filename}')
        img = Image.open(f'{self.file_dest}/{filename}')
        return img

    def capture_stream(self, num_frames=12):
        # function returns a list of images
        # param num_frames: int value with number of frames to capture
        # return frames: images is a list containing jpg images
        frames = []
        start_time = time.time()
        # self.camera2.capture_files(name=self.file_dest+'/stream{:d}.jpg',
        #                            num_files=num_frames, capture_mode='still')
        for image_num in range(num_frames):
            img = self.capture_image_with_file(filename=f'{self.file_dest}/stream{image_num:d}.jpg')
            # img = Image.open(f'{self.file_dest}/stream{image_num:d}.jpg')
            frames.append(img)
        self.FPS = num_frames / float(time.time() - start_time)
        return frames

    # once first image is captured call motion detector in a loop to find each subsequent image
    # motion detection, compute the absolute difference between the current frame and first frame
    # if the difference is more than the tolerance we have something new in the frame aka motion
    def detect(self):
        try:  # trap any camera or image errors gracefully
            self.img = self.capture_image_with_file(filename='capture.jpg')
            grayimg = image_proc.grayscale(self.img)  # convert image to gray scale
            grayblur = image_proc.gaussianblur(grayimg)  # smooth out image for motion detection
            imgdelta = image_proc.compare_images(self.first_img, grayblur)
            self.motion = (self.image_entropy(imgdelta) >= self.min_area)
        except Exception as e:
            self.motion = False
            print(e)
        return self.motion

    def stop(self):
        self.camera2.close()
        return

    # determine change between static image and new frame
    def image_entropy(self, image_delta):
        histogram = image_delta.histogram()
        histlength = sum(histogram)
        probability = [float(h) / histlength for h in histogram]
        return -sum([p * math.log(p, 2) for p in probability if p != 0])


# if __name__ == '__main__':
