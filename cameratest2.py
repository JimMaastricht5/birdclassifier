from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.start_preview()
sleep(2)
camera.capture('testcap.jpg')
camera.stop_preview()