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
from collections import defaultdict


# default dictionary returns a tuple of zero confidence and zero bird count
def default_value():
    return 0


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
    motion_detect = motion_detector.MotionDetector(args=args)  # init class
    print('Done with camera init... setting up classes.')
    bird_tweeter = tweeter.TweeterClass()  # init tweeter2 class twitter handler
    chores = dailychores.DailyChores(bird_tweeter, birdpop, cityweather)
    # init detection and classifier object
    birds = label_image.DetectClassify(homedir=args.homedir, labels=args.labels, thresholds=args.thresholds,
                                       default_confidence=args.default_confidence,
                                       mismatch_penalty=args.mismatch_penalty,
                                       screenheight=args.screenheight, screenwidth=args.screenwidth,
                                       color_chg=args.color_chg,
                                       contrast_chg=args.contrast_chg, sharpness_chg=args.sharpness_chg,
                                       brightness_chg=args.brightness_chg,
                                       overlap_perc_tolerance=args.overlap_perc_tolerancek,
                                       target_object='bird', target_object_min_confidence=.8)
    print('Starting while loop until sun set..... ')
    # loop while the sun is up, look for motion, detect birds, determine species
    while cityweather.sunrise.time() < datetime.now().time() < cityweather.sunset.time():
        if args.verbose:
            chores.hourly_and_daily(filename='')  # perform chores that take on a schedule such as weather reporting
        motion_detect.detect()
        if motion_detect.motion:
            motioncnt += 1

        if motion_detect.motion and birds.detect(img=motion_detect.img):  # daytime with motion and birds
            motioncnt = 0  # reset motion count between detected birds
            print(f'\r Saw a bird {datetime.now().strftime("%I:%M:%S %P")}', end=' ')  # indicate motion on monitor
            # keep first shot to add to start of animation or as stand along jpg
            # classify, grab labels, enhance the shot, and add boxes
            first_img_jpg = birds.img
            if birds.classify(img=first_img_jpg) >= args.default_confidence:  # found a bird we can classify
                first_img_jpg = image_proc.enhance_brightness(img=first_img_jpg, factor=args.brightness_chg)
                birds.set_colors()  # set new colors for this series of bounding boxes
                first_img_jpg = birds.add_boxes_and_labels(img=first_img_jpg)
                gif, gif_filename, animated, best_label, best_confidence = build_bird_animated_gif(args, motion_detect,
                                                                                                   birds, first_img_jpg)
                birdpop.visitors(best_label, datetime.now())  # update census count and time last seen
                bird_first_time_seen = birdpop.first_time_seen
                tweet_label = tweet_text(best_label, best_confidence)
                if animated and bird_first_time_seen:  # doesn't change last_tweet time or override time between tweets
                    print(f'--- First time seeing a {best_label} today.  Tweeting still shot')
                    first_img_jpg.save('first_img.jpg')
                    bird_tweeter.post_image_from_file(message=f'First time today: {tweet_label}',
                                                      file_name='first_img.jpg')
                waittime = birdpop.report_single_census_count(best_label) * args.tweetdelay / 10  # wait X min * N bird
                waittime = args.tweetdelay if waittime >= args.tweetdelay else waittime
                if animated and ((datetime.now() - last_tweet).total_seconds() >= waittime or bird_first_time_seen):
                    print('--- Tweet animated gif at:', datetime.now())
                    if bird_tweeter.post_image_from_file(tweet_label, gif_filename) is False:  # animated gif
                        print(f"*** Failed gif tweet")  # failure, don't update last tweet time
                    else:
                        last_tweet = datetime.now()  # update last tweet time if successful
    print('')  # ending process for evening, print blank line and begin shut down
    motion_detect.stop()
    if args.verbose:
        chores.hourly_and_daily(report_pop=True)
        chores.end_report()  # post a report on run time of the process


def convert_to_list(input_str_list):
    output_list = []
    if type(input_str_list) != list:
        output_list.append(input_str_list)
    else:
        output_list = input_str_list
    return output_list


# should be passing default dictionary
def build_dict(label_dict, input_labels_list, conf_dict, input_confidences_list):
    labels_list = convert_to_list(input_labels_list)
    confidences_list = convert_to_list(input_confidences_list)
    for ii, label in enumerate(labels_list):
        label_dict[label] += 1
        conf_dict[label] += confidences_list[ii]
    return label_dict, conf_dict


# loop thru keys and remove census entries with 1 or zero observations
def remove_single_observations(census_dict, conf_dict):
    key_list = []
    [key_list.append(key) if census_dict[key] <= 1 else '' for key in census_dict]
    [census_dict.pop(key) for key in key_list]
    [conf_dict.pop(key) for key in key_list]
    return census_dict, conf_dict


def best_confidence_and_label(census_dict, confidence_dict):
    best_confidence, best_confidence_label, best_census, best_census_label = 0, '', 0, ''
    try:
        census_dict, confidence_dict = remove_single_observations(census_dict, confidence_dict)  # multishots results
        best_confidence = confidence_dict[max(confidence_dict, key=confidence_dict.get)] / \
            census_dict[max(confidence_dict, key=confidence_dict.get)]  # sum conf/bird cnt
        best_confidence_label = max(confidence_dict, key=confidence_dict.get)
        best_census = census_dict[max(census_dict, key=census_dict.get)]
        best_census_label = max(census_dict, key=census_dict.get)
    except Exception as e:
        print(e)
    print('best confidence:', best_confidence, best_confidence_label)
    print('best census:', best_census, best_census_label)
    # if best_confidence_label != best_census_label:
    #  what to do here?  Which one is better?  High count or high confidence?
    return best_confidence, best_confidence_label, best_census, best_census_label


def build_bird_animated_gif(args, motion_detect, birds, first_img_jpg):
    # grab a stream of pictures, add first pic from above, and build animated gif
    # return gif, filename, animated boolean, and best label as the max of all confidences
    gif_filename, best_label, best_confidence, labeled_frames = '', '', 0, []
    animated = False  # set to true if min # of frames captured with birds
    gif = first_img_jpg  # set a default if animated = False
    last_good_frame = 0  # find last frame that has a bird, index zero is good based on first image
    frames_with_birds = 1  # count of frames with birds, set to 1 for first img
    census_dict = defaultdict(default_value)  # track all results and pick best confidence
    confidence_dict = defaultdict(default_value)  # track all results and pick best confidence
    census_dict, confidence_dict = build_dict(census_dict, birds.classified_labels, confidence_dict,
                                              birds.classified_confidences)
    frames = motion_detect.capture_stream()  # capture a list of images
    labeled_frames.insert(0, image_proc.convert_image(img=first_img_jpg, target='gif'))  # isrt 1st img
    for i, frame in enumerate(frames):
        frame = image_proc.enhance_brightness(img=frame, factor=args.brightness_chg)
        if birds.detect(img=frame):  # find bird object in frame and set rectangles containing object
            if birds.classify(img=frame, use_confidence_threshold=False) > 0:   # classify at rectangle & chk confidence
                frames_with_birds += 1
                last_good_frame = i + 1  # found a bird, add one to last good frame to account for insert of 1st image
            census_dict, confidence_dict = build_dict(census_dict, birds.classified_labels, confidence_dict,
                                                      birds.classified_confidences)
        labeled_frames.append(birds.add_boxes_and_labels(img=frame, use_last_known=True))  # append image regardless
    if frames_with_birds >= (args.minanimatedframes - 1):  # if bird is in min number of frames build gif
        gif, gif_filename = image_proc.save_gif(frames=labeled_frames[0:last_good_frame], frame_rate=args.framerate)
        animated = True
        _, _, _, _ = best_confidence_and_label(census_dict, confidence_dict)
        best_confidence = confidence_dict[max(confidence_dict, key=confidence_dict.get)] / \
            census_dict[max(confidence_dict, key=confidence_dict.get)]  # sum conf/bird cnt
        best_label = max(confidence_dict, key=confidence_dict.get)
        print('--- Best label and confidence', best_label, best_confidence)
    return gif, gif_filename, animated, best_label, best_confidence


def tweet_text(label, confidence):
    # sample url https://www.allaboutbirds.org/guide/Northern_Rough-winged_Swallow/overview
    sname = str(label)  # make sure the label is a string
    sname = sname[sname.find(' ') + 1:] if sname.find(' ') >= 0 else sname  # remove index number
    cname = sname[sname.find('(') + 1: sname.find(')')] if sname.find('(') >= 0 else sname  # retrieve common name
    hypername = cname.replace(' ', '_')
    hyperlink = f'https://www.allaboutbirds.org/guide/{hypername}/overview'
    tweet_label = f'{cname} {confidence * 100:.1f}% {hyperlink}'
    return tweet_label


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    # camera settings
    ap.add_argument("-fc", "--flipcamera", type=bool, default=False, help="flip camera image")
    ap.add_argument("-sw", "--screenwidth", type=int, default=640, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=480, help="max screen height")
    ap.add_argument("-fr", "--framerate", type=int, default=45, help="frame rate for camera")
    ap.add_argument("-gf", "--minanimatedframes", type=int, default=10, help="minimum number of frames with a bird")
    ap.add_argument("-st", "--save_img", type=bool, default=False, help="write images to disk")
    ap.add_argument("-v", "--verbose", type=bool, default=True, help="To tweet extra stuff or not")
    ap.add_argument("-td", "--tweetdelay", type=int, default=3600,
                    help="Wait time between tweets is N species seen delay/10 with not to exceed max of tweet delay")

    # motion and image processing settings
    ap.add_argument("-b", "--brightness_chg", type=int, default=1.05, help="brightness boost")  # 1 no chg,< 1 -, > 1 +
    ap.add_argument("-c", "--contrast_chg", type=float, default=1.2, help="contrast boost")  # 1 no chg,< 1 -, > 1 +
    ap.add_argument("-cl", "--color_chg", type=float, default=1.2, help="color boost")  # 1 no chg,< 1 -, > 1 +
    ap.add_argument("-sp", "--sharpness_chg", type=float, default=1.2, help="sharpeness")  # 1 no chg,< 1 -, > 1 +
    ap.add_argument("-mp", "--mismatch_penalty", type=float, default=.3,
                    help="confidence penalty if predictions from img and enhance img dont match ")
    ap.add_argument("-ei", "--enhanceimg", type=bool, default=True, help="offset waterproof box blur and enhance img")
    ap.add_argument("-co", "--default_confidence", type=float, default=.98, help="confidence threshold")
    ap.add_argument("-op", "--overlap_perc_tolerance", type=float, default=0.8, help="% box overlap to flag as dup")
    ap.add_argument("-ma", "--minarea", type=float, default=5.0, help="motion entropy threshold")  # lower = > motion
    ap.add_argument("-hd", "--homedir", type=str, default='/home/pi/PycharmProjects/birdclassifier/',
                    help="home directory for files")
    ap.add_argument("-la", "--labels", type=str, default='coral.ai.inat_bird_labels.txt',
                    help="name of file to use for species labels and thresholds")
    ap.add_argument("-tr", "--thresholds", type=str, default='coral.ai.inat_bird_threshold.csv',
                    help="name of file to use for species labels and thresholds")

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
