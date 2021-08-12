from setuptools import setup
# PCA9685 needs to get added for pan tilt if you are using that.
# tensorflow package is required for windows, tflite-runtime is the correct lib for rasp pi

# from https://www.tensorflow.org/lite/guide/python
# If you're running Debian Linux or a derivative of Debian (including Raspberry Pi OS),
# you should install from our Debian package repo. This requires that you add a new repo list
# and key to your system and then install as follows:
# echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
# curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
# sudo apt-get update
# sudo apt-get install python3-tflite-runtime

setup(
    name='Bird Class Python App',
    version='1.0',
    packages=['cv2, scipy, requests, imutils, opencv-python, numpy, oauthlib, tflite-runtime,'
              'tywthon, Pillow, argparse, datetime, logging, PIL, scikit-image'],
    url='',
    license='',
    author='maastricht',
    author_email='jimmaastricht5@gmail.com',
    description='set of python lib for image detection and classificatio of birds; allows for ' +
                'motion tracking with pan/tilt. PCA9685 only required if using pantilt on Pi'
)
