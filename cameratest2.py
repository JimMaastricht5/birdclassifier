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

from picamera2 import Picamera2
from time import sleep
import os

screenheight = 640
screenwidth = 480

camera = Picamera2()
camera.vflip = False

# Configure preview resolution
config = Picamera2.preview.preview_configuration(
    size=(screenwidth, screenheight)
)

# Create preview stream (optional, comment out if not needed)
# preview_stream = camera.create_preview_stream(config)
# preview_stream.start()  # Uncomment to show preview
sleep(2)  # Adjust sleep duration as needed

# Create capture request
capture_request = camera.create_still_capture_request(config)

# Capture image
camera.capture(capture_request)

# Save captured image
with camera.capture_file(os.getcwd() + '/assets/testcap2.jpg') as file:
    file.write(camera.capture_buffer)

# Information printing (modify as needed based on picamera2 documentation)
print("Frame rate range:", camera.framerate_range)  # Might require different method
# ... Consult documentation for other properties ...

# Stop preview if started
# preview_stream.stop()

camera.close()  # Close camera resources




