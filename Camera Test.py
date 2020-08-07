# https://www.pyimagesearch.com/2019/09/16/install-opencv-4-on-raspberry-pi-4-and-raspbian-buster/
# pip3 install opencv-contrib-python==4.1.0.25
# Q: What do I do when I encounter an "undefined symbol: __atomic_fetch_add8" error related to libatomic.so?
# A: The pip install has been giving readers troubles since OpenCV 4.1.1
# (around the November 2019 timeframe). Be sure to install version 4.1.0.25 :
# $ pip install opencv-contrib-python==4.1.0.25
# set includ-system-site-packages = true or run pip commands in pycharm terminal

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
print('press q to end')

while (True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, -1)  # Flip camera vertically
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()