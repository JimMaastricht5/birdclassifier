from setuptools import setup

setup(
    name='Bird Class Python App',
    version='1.0',
    packages=['imutils, opencv-python, numpy, oauthlib, tensorflow, tywthon, Pillow, argparse, datetime, logging, PIL, scikit-image, PCA9685'],
    url='',
    license='',
    author='maastricht',
    author_email='jimmaastricht5@gmail.com',
    description='set of python lib for image detection and classificatio of birds; allows for ' +
                'motion tracking with pan/tilt. PCA9685 only required if using pantilt on Pi'
)
