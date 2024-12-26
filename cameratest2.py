# MIT License
#
# 2024 Jim Maastricht
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import os
import argparse  # argument parser
from time import sleep
from picamera2 import Picamera2, Preview
from libcamera import Transform  # add import to test setup for motion detector


def camera_test(args):
    """ function tests camera on rasp pi
    :param args: command line arguments, include screenheight, screenwidth, and flipcamera (vertical)
    :return: Nothing
    """
    picam2 = Picamera2()

    config = picam2.create_preview_configuration(main={"size": (args.screenheight, args.screenwidth)},
                                                 transform=Transform(vflip=args.flipcamera))
    picam2.configure(config)
    picam2.start_preview()
    picam2.start(Preview.QT)
    sleep(args.sleep_time)  # let the camera settle
    metadata = picam2.capture_file(args.directory + '/testcap2.jpg')
    print(metadata)

    picam2.close()
    return


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    # load settings from config file to allow for simple override
    ap = argparse.ArgumentParser()
    # camera settings
    ap.add_argument("-fc", "--flipcamera", type=bool, default=False, help="flip camera image")
    ap.add_argument("-sw", "--screenwidth", type=int, default=640, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=480, help="max screen height")

    # general app settings
    ap.add_argument("-st", "--sleep_time", type=int, default=2, help="time to hold preview")
    ap.add_argument("-dr", "--directory", type=str, default="/home/pi/birdclassifier/assets",
                    help="directory to write test file")

    camera_test(ap.parse_args())
