from setuptools import setup
# PCA9685 needs to get added for pan tilt if you are using that.

# tensorflow package is required for windows, tflite-runtime is the correct lib for rasp pi
# PY4: pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
# or try....
# from https://www.tensorflow.org/lite/guide/python
# If you're running Debian Linux or a derivative of Debian (including Raspberry Pi OS),
# you should install from our Debian package repo. This requires that you add a new repo list
# and key to your system and then install as follows:
# echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
# curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
# sudo apt-get update
# sudo apt-get install python3-tflite-runtime


# to install picamera on windows for dev
# Open terminal (if you have a virtual env, activate it)
# type "set READTHEDOCS=True"
# pip install picamera OR 4.pip install picamera[array]

setup(
    name='Bird Class Python App',
    version='1.0',
    packages=['scipy, requests, numpy, oauthlib, picamera'
              'tywthon, Pillow, argparse, datetime, logging, scikit-image'],
    url='',
    license='',
    author='maastricht',
    author_email='jimmaastricht5@gmail.com',
    description='set of python lib for image detection and classificatio of birds; allows for ' +
                'motion tracking with pan/tilt. PCA9685 only required if using pantilt on Pi'
)
