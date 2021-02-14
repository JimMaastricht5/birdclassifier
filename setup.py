from setuptools import setup
# PCA9685 needs to get added for pan tilt if you are using that.

setup(
    name='Bird Class Python App',
    version='1.0',
    packages=['imutils, opencv-python, numpy, oauthlib, tensorflow, tflite-runtime, tywthon, Pillow, argparse, datetime, logging, PIL, scikit-image'],
    url='',
    license='',
    author='maastricht',
    author_email='jimmaastricht5@gmail.com',
    description='set of python lib for image detection and classificatio of birds; allows for ' +
                'motion tracking with pan/tilt. PCA9685 only required if using pantilt on Pi'
)
