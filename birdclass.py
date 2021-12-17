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
# packages: twitter use twython package, auth.py must be in project for import auth
#   oauthlib,
import image_proc
import label_image  # code to init tensor flow model and classify bird type, bird object
import motion_detector  # motion detector helper functions
import objtracker  # keeps track of detected objects between frames
import tweeter  # twitter helper functions
import population  # population census object, tracks species total seen and last time
import dailychores  # handles tasks that occur once per day or per hour
import weather
import argparse  # argument parser
from datetime import datetime
import time


def bird_detector(args):
    birdpop = population.Census()  # initialize species population census object
    birdobj = objtracker.CentroidTracker()
    motioncnt = 0
    curr_day, curr_hr, last_tweet = datetime.now().day, datetime.now().hour, datetime(2021, 1, 1, 0, 0, 0)
    cityweather = weather.City_Weather()  # init class and set var based on default of Madison WI

    print(f'It is now {datetime.now()}.  \nSunrise at {cityweather.sunrise} and sunset at {cityweather.sunset}.')
    # wait here until the sun is up before initialize the camera
    if datetime.now().time() < cityweather.sunrise.time():
        waittime = (cityweather.sunrise - datetime.now()).total_seconds()
        print(f'taking a {waittime} second to wait for sun rise')
        time.sleep(waittime)  # wait until the sun comes up

    # initial video capture, screen size, and grab first image (no motion)
    camera, first_img = motion_detector.init(args)  # set gray motion mask
    print('done with camera init... setting up classes.')
    bird_tweeter = tweeter.Tweeter_Class()  # init tweeter2 class twitter handler
    chores = dailychores.DailyChores(bird_tweeter, birdpop, cityweather)
    # init detection and classifier object
    birds = label_image.DetectClassify(default_confidence=args.default_confidence,
                                       mismatch_penalty=args.mismatch_penalty,
                                       screenheight=args.screenheight, screenwidth=args.screenwidth,
                                       framerate=args.framerate, color_chg=args.color_chg,
                                       contrast_chg=args.contrast_chg, sharpness_chg=args.sharpness_chg,
                                       overlap_perc_tolerance=args.overlap_perc_tolerance)

    # camera.start_preview()  # lets see what is going on....
    print('starting while loop until sun set..... ')
    while True:  # look for motion, detect birds, and determine species; break at end of day
        chores.hourly_and_daily()  # perform chores that take place hourly or daily such as weather reporting
        motionb, img = motion_detector.detect(camera, first_img, args.minarea)
        if motionb is True:  # motion but no birds
            motioncnt += 1
            print(f'\r motion {motioncnt}', end=' ')  # indicate motion on monitor

        if motionb and birds.detect(img):  # daytime with motion and birds
            motioncnt = 0  # reset motion count between detected birds
            print('')  # print new lines between birds detection for motion counter
            birds.classify()
            birdobj.update(birds.classified_rects, birds.classified_confidences, birds.classified_labels)
        else:  # no birds detected in frame or not daytime, update missing from frame count
            birdobj.update_null()
            birds.target_object_found = False  # may not have run motion detection if not daylight

        if birds.target_object_found is True:  # saw at least one bird
            common_names, tweet_label = label_text(birds.classified_labels, birds.classified_confidences)
            if args.enhanceimg:
                birds.equalizedimg = birds.add_boxes_and_labels(birds.equalizedimg, common_names,
                                                                birds.classified_rects)
                # place holder to show predicted birds.equalizedimg
            else:
                birds.img = image_proc.enhance_brightness(birds.img, factor=args.brightness)
                birds.img = birds.add_boxes_and_labels(birds.img, common_names, birds.classified_rects)
                # place holder to show predicted birds.img

            # Show image and tweet, confidence
            if all(conf >= birds.classify_min_confidence for conf in birds.classified_confidences):  # tweet threshold
                if (datetime.now() - last_tweet).total_seconds() >= 60 * 5:
                    birdpop.visitors(birds.classified_labels, datetime.now())  # update census count and last tweeted
                    last_tweet = datetime.now()
                    if args.enhanceimg:  # decide what to tweet
                        # place holder to show tweeted birds.equalizedimg
                        tweetedb = bird_tweeter.post_image(tweet_label, birds.equalizedimg)
                    else:
                        # place holder to show tweeted birds.img
                        tweetedb = bird_tweeter.post_image(tweet_label, birds.img)
                    if tweetedb is False:
                        print(f"*** exceeded tweet limit")
                else:
                    # print(f" {tweet_label} not tweeted, last tweet {last_tweet.strftime('%I:%M %p')}. wait 5 minutes")
                    pass

        # motion processed, all birds in image processed if detected, add all known objects to image
        # try:
        #    birds.img = birds.add_boxes_and_labels(birds.img, birdobj.objnames, birdobj.rects)
        # except:
        #    print('*** error in boxes and labels using image tracker')
        # if birds.target_object_found:
        #     print('*** bird detect and classify results')
        #     print(birds.classified_labels, birds.classified_rects)
        # place holder show video w boxes and labels

        # shut down the app after sunset
        if datetime.now().time() >= cityweather.sunset.time():
            break

    # camera.stop_preview()
    camera.close()
    chores.hourly_and_daily(report_pop=True)
    chores.end_report()  # post a report on run time of the process


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
        common_names += f'{cname} {species_confs[i] * 100:.1f}%'
        tweet_label += f'{sname} {species_confs[i] * 100:.1f}%'
    return common_names, tweet_label


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    # camera settings
    ap.add_argument("-fc", "--flipcamera", type=bool, default=False, help="flip camera image")
    ap.add_argument("-sw", "--screenwidth", type=int, default=640, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=480, help="max screen height")
    ap.add_argument("-fr", "--framerate", type=int, default=30, help="frame rate for camera")

    # motion and image processing settings
    ap.add_argument("-b", "--brightness_chg", type=int, default=1.05, help="brightness boost")  # 1 no chg,< 1 -, > 1 +
    ap.add_argument("-c", "--contrast_chg", type=float, default=1.2, help="contrast boost")  # 1 no chg,< 1 -, > 1 +
    ap.add_argument("-cl", "--color_chg", type=float, default=1.2, help="color boost")  # 1 no chg,< 1 -, > 1 +
    ap.add_argument("-sp", "--sharpness_chg", type=float, default=1.2, help="sharpeness")  # 1 no chg,< 1 -, > 1 +
    ap.add_argument("-mp", "--mismatch_penalty", type=float, default=.3,
                    help="confidence penalty if predictions from img and enhance img dont match ")
    ap.add_argument("-ei", "--enhanceimg", type=bool, default=True, help="offset waterproof box blur and enhance img")
    ap.add_argument("-co", "--default_confidence", type=float, default=.95, help="confidence threshold")
    ap.add_argument("-op", "--overlap_perc_tolerance", type=float, default=0.6, help="% box overlap to flag as dup")
    ap.add_argument("-ma", "--minarea", type=float, default=5.50, help="motion entropy threshold")  # < no motion

    arguments = ap.parse_args()
    bird_detector(arguments)
