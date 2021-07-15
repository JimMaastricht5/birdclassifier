# MIT License
#
# 2021 Jim Maastricht
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
# motion detector with builtin bird detection and bird classification
# built by JimMaastricht5@gmail.com
# uses tflite prebuilt google model for object detection and tensorflow lite model for bird classification
# currently using a model from coral.ai for species identification.
# Other species models built with tensor flow  using tools from tensor in colab notebook
# https://colab.research.google.com/drive/1taZ9JincTaZuZh_JmBSC4pAbSQavxbq5#scrollTo=D3i_6WSXjUhk
# special notes when setting up the code on a rasberry pi 4 9/9/20
# install supporting libraries for directions @: https://qengineering.eu/install-opencv-4.2-on-raspberry-pi-4.html
# install OpenCv version 4.4+
# packages: twitter use twython package, auth.py must be in project for import auth
#   oauthlib,
import cv2  # open cv 2
import label_image  # code to init tensor flow model and classify bird type
import motion_detector  # motion detector helper functions
import objtracker  # keeps track of detected objects between frames
import weather
import tweeter  # twitter helper functions
import population  # population census object, tracks species total seen and last time
import argparse  # argument parser
from datetime import datetime
from gpiozero import CPUTemperature
from subprocess import call


def bird_detector(args):
    birdpop = population.Census()  # initialize species population census object
    birdobj = objtracker.CentroidTracker()
    motioncnt = 0
    curr_day, curr_hr, last_tweet = datetime.now().day, datetime.now().hour, datetime(2021, 1, 1, 0, 0, 0)
    spweather = weather.City_Weather()  # init class and set var based on default of Madison WI

    # initial video capture, screen size, and grab first image (no motion)
    cap = cv2.VideoCapture(0)  # capture video image
    # cap.set(3, args.screenwidth)  # set screen width
    # cap.set(4, args.screenheight)  # set screen height
    first_img = motion_detector.init(args.flipcamera, cv2, cap)  # set gray motion mask
    set_windows()  # position output windows at top of screen and init output

    # setup twitter and tensor flow models
    bird_tweeter = tweeter.Tweeter_Class()  # init tweeter2 class twitter handler
    birds = label_image.DetectClassify()  # init detection and classifier object
    starttime = datetime.now()  # used for total run time report
    bird_tweeter.post_status(f'Starting process at {datetime.now().strftime("%I:%M:%S %P")}, ' +
                             f'{spweather.weatherdescription} ' +
                             f'with {spweather.skycondition}% cloud cover. Visibility of {spweather.visibility} ft.' +
                             f' Temp is currently {spweather.temp}F with ' +
                             f'wind speeds of {spweather.windspeed} MPH.')

    while True:  # while escape key is not pressed look for motion, detect birds, and determine species
        curr_day, curr_hr = hour_or_day_change(curr_day, curr_hr, spweather, bird_tweeter, birdpop)
        motionb, img = motion_detector.detect(args.flipcamera, cv2, cap, first_img, args.minarea)
        cv2.imshow('video', img)  # show video w no boxes or labels use cv2.flip if image inverted
        if motionb is True and birds.detect(img):  # motion with birds
            motioncnt = 0  # reset motion count between detected birds
            print('')  # print new lines between birds detection for motion counter
            birds.classify()
            birdobj.update(birds.classified_rects, birds.classified_confidences, birds.classified_labels)
        else:  # no birds detected in frame, update missing from frame count
            birdobj.update_null()
            if motionb is True:  # motion but no birds
                motioncnt += 1
                print(f'\r motion {motioncnt}', end=' ')  # indicate motion on monitor

        if birds.target_object_found is True:  # saw at least one bird
            common_names, tweet_label = label_text(birds.classified_labels, birds.classified_confidences)
            birds.equalizedimg = birds.add_boxes_and_labels(birds.equalizedimg, common_names, birds.classified_rects)
            # birds.img = birds.add_boxes_and_labels(birds.img, common_names, birds.classified_rects)
            if args.enhanceimg:
                cv2.imshow('predicted', birds.equalizedimg)  # show equalized image
            else:
                cv2.imshow('predicted', birds.img)

            # Show image and tweet, confidence
            if all(conf >= birds.classify_min_confidence for conf in birds.classified_confidences):  # tweet threshold
                if (datetime.now() - last_tweet).total_seconds() >= 60 * 5:
                    birdpop.visitors(birds.classified_labels, datetime.now())  # update census count and last tweeted
                    last_tweet = datetime.now()
                    # decide what to tweet
                    if args.enhanceimg:
                        cv2.imshow('tweeted', birds.equalizedimg)
                        tweetedb = bird_tweeter.post_image(tweet_label, birds.equalizedimg)
                    else:
                        cv2.imshow('tweeted', birds.img)
                        tweetedb = bird_tweeter.post_image(tweet_label, birds.img)
                    if tweetedb == False:
                        print(f"*** exceeded tweet limit")
                else:
                    print(f" {tweet_label} not tweeted, last tweet {last_tweet.strftime('%I:%M %p')}. wait 5 minutes")

        # motion processed, all birds in image processed if detected, add all known objects to image
        try:
            birds.img = birds.add_boxes_and_labels(birds.img, birdobj.objnames, birdobj.rects)
        except:
            print('*** error in boxes and labels using image tracker')
        # if birds.target_object_found:
        #     print('*** bird detect and classify results')
        #     print(birds.classified_labels, birds.classified_rects)
        cv2.waitKey(20)  # wait 20 ms to render video, restart loop.  setting of 0 is fixed img; > 0 video
        # shut down the app if between 1:00 and 1:05 am.  Pi runs this in a loop and restarts it every 20 minutes
        if datetime.now().hour == 1 and datetime.now().minute <= 5:
            break

    # while loop break at 10pm, shut down windows
    cap.release()
    cv2.destroyAllWindows()
    bird_tweeter.post_status(f'Ending process at {datetime.now().strftime("%I:%M:%S %P")}.  Run time was ' +
                             f'{divmod((datetime.now() - starttime).total_seconds(), 60)[0]} minutes')


# housekeeping for day and hour
def hour_or_day_change(curr_day, curr_hr, spweather, bird_tweeter, birdpop):
    post_txt = ''  # force to string
    if curr_day != datetime.now().day:
        observed = birdpop.get_census_by_count()  # count from prior day
        post_txt = f'top 3 birds for day {str(curr_day)}'
        index, loopcnt = 0, 1
        while loopcnt <= 3:  # top 3 skipping unknown species
            birdstr = ''  # used to force tuple to string
            if observed[index][0:2] == '':
                index += 1
            try:
                birdstr = f', #{str(loopcnt)} {observed[index][0:2]}'
            except IndexError:
                break
            post_txt = post_txt + birdstr
            index += 1
            loopcnt += 1

        bird_tweeter.post_status(post_txt[0:150])
        birdpop.clear()  # clear count for new day
        curr_day = datetime.now().day  # set new day = to current day

    if curr_hr != datetime.now().hour:  # check weather and CPU temp hourly
        spweather.update_conditions()
        curr_hr = datetime.now().hour
        cpu = CPUTemperature()
        print(f'***hourly temp check. cpu temp is: {cpu.temperature}C {(cpu.temperature * 9/5) + 32}F')
        try:
            if int(cpu.temperature) >= 86:  # limit is 85 C
                bird_tweeter.post_status(f'***shut down. temp: {cpu.temperature}')
                call("sudo shutdown -poweroff")
        except:
            pass
    return curr_day, curr_hr


# set label for image and tweet, use short species name instead of scientific name
def label_text(species_names, species_confs):
    common_names, tweet_label, cname, sname = '', '', '', ''
    for i, sname in enumerate(species_names):
        if i > 0:
            common_names += ', '
            tweet_label += ', '

        sname = str(sname)  # make sure species is considered a string
        start = sname.find('(') + 1  # find start of common name, move one character to drop (
        end = sname.find(')')
        if start >= 0 and end >= 0:
            cname = sname[start:end]
        else:
            cname = sname
        common_names = common_names + cname
        tweet_label += sname + ' ' + f'{species_confs[i] * 100:.1f}%'
    return common_names, tweet_label


def set_windows():
    cv2.namedWindow('video')
    cv2.namedWindow('predicted')
    cv2.namedWindow('tweeted')

    cv2.moveWindow('video', 0, 0)
    cv2.moveWindow('predicted', 350, 0)
    cv2.moveWindow('tweeted', 700, 0)

    cv2.waitKey(20)  # wait 20 ms to render video, restart loop.  setting of 0 is fixed img; > 0 video
    return


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--flipcamera", type=bool, default=False, help="flip camera image")
    ap.add_argument("-e", "--enhanceimg", type=bool, default=False, help="flip camera image")
    ap.add_argument("-a", "--minarea", type=int, default=1000, help="motion threshold")
    ap.add_argument("-sw", "--screenwidth", type=int, default=320, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=240, help="max screen height")

    arguments = ap.parse_args()
    bird_detector(arguments)
