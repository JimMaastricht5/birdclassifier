from picamera import PiCamera
from time import sleep
import os

# screenheight = 720
# screenwidth = 640
camera = PiCamera()
camera.vflip = False
# camera.resolution = (screenheight, screenwidth)
camera.start_preview()
sleep(2)
camera.capture(os.getcwd()+ '/assets/testcap2.jpg')
camera.stop_preview()
