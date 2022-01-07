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
import tweeter  # twitter helper functions
import population  # population census object, tracks species total seen and last time
import dailychores  # handle tasks that occur once per day or per hour
import weather
import argparse  # argument parser
from datetime import datetime


def bird_detector(args):
    birdpop = population.Census()  # initialize species population census object
    motioncnt = 0
    curr_day, curr_hr, last_tweet = datetime.now().day, datetime.now().hour, datetime(2021, 1, 1, 0, 0, 0)

    # while loop below processes from sunrise to sunset.  The python program runs in a bash loop
    # that restarts itself after downloading a new version of this software
    # we want to wait to enter that main while loop until sunrise
    cityweather = weather.CityWeather()  # init class and set var based on default of Madison WI
    print(f'It is now {datetime.now()}.  \nSunrise at {cityweather.sunrise} and sunset at {cityweather.sunset}.')
    cityweather.wait_until_midnight()  # if after sunset, wait here until after midnight
    cityweather.wait_until_sunrise()  # if before sun rise, wait here

    # initial video capture, screen size, and grab first image (no motion)
    motion_detect = motion_detector.MotionDetector(args=args, save_test_img=args.save_test_img)  # init class
    print('done with camera init... setting up classes.')
    bird_tweeter = tweeter.Tweeter_Class()  # init tweeter2 class twitter handler
    chores = dailychores.DailyChores(bird_tweeter, birdpop, cityweather)
    # init detection and classifier object
    birds = label_image.DetectClassify(default_confidence=args.default_confidence,
                                       mismatch_penalty=args.mismatch_penalty,
                                       screenheight=args.screenheight, screenwidth=args.screenwidth,
                                       color_chg=args.color_chg,
                                       contrast_chg=args.contrast_chg, sharpness_chg=args.sharpness_chg,
                                       overlap_perc_tolerance=args.overlap_perc_tolerance)

    print('starting while loop until sun set..... ')
    # loop while the sun is up, look for motion, detect birds, determine species
    while cityweather.sunrise.time() < datetime.now().time() < cityweather.sunset.time():
        if args.verbose:
            chores.hourly_and_daily()  # perform chores that take place hourly or daily such as weather reporting

        motion_detect.detect()
        if motion_detect.motion:
            motioncnt += 1
            print(f'\r motion {motioncnt}', end=' ')  # indicate motion on monitor

        if motion_detect.motion and birds.detect(img=motion_detect.img):  # daytime with motion and birds
            motioncnt = 0  # reset motion count between detected birds

            # keep first shot to add to start of animation or as stand along jpg
            # copy first image, classify, grab labels, enhance the shot, and add boxes
            first_img_jpg = birds.img.copy()
            first_img_confidence = birds.classify(img=first_img_jpg)
            first_tweet_label = tweet_text(birds.classified_labels, birds.classified_confidences)
            first_img_jpg = image_proc.enhance_brightness(img=first_img_jpg, factor=args.brightness_chg)
            birds.pick_a_color()  # set new color for this series of bounding boxes
            first_img_jpg = birds.add_boxes_and_labels(img=first_img_jpg)
            birdpop.visitors(birds.classified_labels, datetime.now())  # update census count and time last seen
            # if found a bird that we're confident in, grab a stream of pics, add first pic, and build animated gif
            if first_img_confidence >= args.default_confidence:
                gif = build_bird_animated_gif(args, motion_detect, birds, first_img_jpg)
                # what to do if bird flies off at first img grab?
                print('ready to tweet, wait five minutes since last tweet.  last tweet was at:', last_tweet)
                if (datetime.now() - last_tweet).total_seconds() >= 60 * 5:
                    last_tweet = datetime.now()
                    if bird_tweeter.post_image(first_tweet_label, gif) is False:  # try animated gif
                        print(f"*** failed gif tweet")
                        if bird_tweeter.post_image(first_tweet_label, first_img_jpg) is False:  # try org jpg
                            print(f"*** failed jpg tweet")

    motion_detect.stop()
    if args.verbose:
        chores.hourly_and_daily(report_pop=True)
        chores.end_report()  # post a report on run time of the process


def build_bird_animated_gif(args, motion_detect, birds, first_img_jpg):
    # grab a stream of pictures, add first pic from above, and build animated gif
    labeled_frames = []
    max_confidences_per_frame = []
    frames = motion_detect.capture_stream(save_test_img=args.save_test_img)  # capture a list of images
    for frame in frames:
        birds.detect(img=frame)  # find bird object in frame and set rectangles containing object
        max_confidences_per_frame.append(birds.classify(img=frame))  # classify object at rectangle location
        frame = image_proc.enhance_brightness(img=frame, factor=args.brightness_chg)
        frame = birds.add_boxes_and_labels(img=frame)
        labeled_frames.append(frame)

    labeled_frames.insert(0, image_proc.convert_image(img=first_img_jpg, target='gif',
                                              save_test_img=args.save_test_img))  # covert and insert first img
    gif = image_proc.save_gif(frames=labeled_frames, frame_rate=args.framerate,
                              save_test_img=args.save_test_img)  # build the labeled gif, default file name
    return gif


def tweet_text(classified_labels, classified_confidences):
    tweet_label, sname = '', ''
    for i, sname in enumerate(classified_labels):
        sname = str(sname)  # make sure label is considered a string
        tweet_label += f'{sname} {classified_confidences[i] * 100:.1f}% '
    return tweet_label


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    # camera settings
    ap.add_argument("-fc", "--flipcamera", type=bool, default=False, help="flip camera image")
    ap.add_argument("-sw", "--screenwidth", type=int, default=640, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=480, help="max screen height")
    ap.add_argument("-fr", "--framerate", type=int, default=60, help="frame rate for camera")
    ap.add_argument("-st", "--save_test_img", type=bool, default=False, help="save test images")  # saves sample images
    ap.add_argument("-v", "--verbose", type=bool, default=True, help="To tweet or not to tweet")

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
    ap.add_argument("-ma", "--minarea", type=float, default=5.0, help="motion entropy threshold")  # lower = > motion

    arguments = ap.parse_args()
    bird_detector(arguments)

# junk code
# import objtracker  # keeps track of detected objects between frames
# birdobj = objtracker.CentroidTracker()
# birdobj.update(birds.classified_rects, birds.classified_confidences, birds.classified_labels)
# birdobj.update_null()
# motion processed, all birds in image processed if detected, add all known objects to image
# try:
#    birds.img = birds.add_boxes_and_labels(birds.img, birdobj.objnames, birdobj.rects)
# except:
#    print('*** error in boxes and labels using image tracker')
# if birds.target_object_found:
#     print('*** bird detect and classify results')
#     print(birds.classified_labels, birds.classified_rects)
# placeholder show video w boxes and labels
