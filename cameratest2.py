from picamera import PiCamera
from time import sleep
import os

camera = PiCamera()
camera.vflip = True
camera.start_preview()
sleep(2)
camera.capture(os.getcwd()+ '/assets/testcap2.jpg')
camera.stop_preview()
