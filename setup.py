from setuptools import setup
# PCA9685 needs to get added for pan tilt if you are using that.
# tensorflow package is required for windows, tflite-runtime is the correct lib for rasp pi

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
