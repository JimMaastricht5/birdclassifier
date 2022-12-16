from picamera import PiCamera
from time import sleep
import os

# 1024 x 768 or 720 x 640
screenheight = 640
screenwidth = 480
camera = PiCamera()
camera.vflip = False
#camera.exposure_mode = 'night'  # sports, off
camera.iso = 800
camera.resolution = (screenheight, screenwidth)
camera.start_preview()
sleep(2)
camera.capture(os.getcwd()+ '/assets/testcap2.jpg')
camera.stop_preview()

print(camera.framerate_range.low)
print(camera.framerate_range.high)
print(camera.shutter_speed)
print(camera.exposure_speed)
print(camera.exposure_mode)



