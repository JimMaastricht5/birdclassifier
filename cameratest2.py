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
# old picamera 1 code
# 1024 x 768 or 720 x 640
# screenheight = 640
# screenwidth = 480
# camera = PiCamera()
# camera.vflip = False
# camera.resolution = (screenheight, screenwidth)
# camera.start_preview()
# sleep(2)
# camera.capture(os.getcwd()+ '/assets/testcap2.jpg')
# camera.stop_preview()
#
# print(camera.framerate_range.low)
# print(camera.framerate_range.high)
# print(camera.shutter_speed)
# print(camera.exposure_speed)
# print(camera.exposure_mode)

from time import sleep
import os
from picamera2 import Picamera2, Preview

screenheight = 640
screenwidth = 480
picam2 = Picamera2()

preview_config = picam2.create_preview_configuration(main={"size": (screenheight, screenwidth)})
picam2.configure(preview_config)
# picam2.start_preview(Preview.QTGL)  # does not work, may need to install qt graphics lib
picam2.start()
sleep(2)  # let the camera settle
metadata = picam2.capture_file(os.getcwd()+ '/assets/testcap2.jpg')
# metadata = picam2.capture_file("test.jpg")  # writes file to current dir
print(metadata)

picam2.close()


