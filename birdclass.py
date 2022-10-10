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
import output_stream
import argparse  # argument parser
from datetime import datetime
from collections import defaultdict
import os


# default dictionary returns a tuple of zero confidence and zero bird count
def default_value():
    return 0


def bird_detector(args):
    favorite_birds = ['Northern Cardinal', 'Rose-breasted Grosbeak', 'American Goldfinch']
    birdpop = population.Census()  # initialize species population census object
    output = output_stream.Controller()  # initialize class to handle terminal and web output
    output.start_stream()  # start streaming to terminal and web
    motioncnt, event_count = 0, 0
    curr_day, curr_hr, last_tweet = datetime.now().day, datetime.now().hour, datetime(2021, 1, 1, 0, 0, 0)

    # while loop below processes from sunrise to sunset.  The python program runs in a bash loop
    # that restarts itself after downloading a new version of this software
    # we want to wait to enter that main while loop until sunrise
    cityweather = weather.CityWeather(city=args.city, units='Imperial', iscloudy=60)  # init class and set vars
    output.message(message=f'Now: {datetime.now()}.  \nSunrise: {cityweather.sunrise} Sunset: {cityweather.sunset}.',
                   msg_type='weather')
    cityweather.wait_until_midnight()  # if after sunset, wait here until after midnight
    cityweather.wait_until_sunrise()  # if before sun rise, wait here

    # initial video capture, screen size, and grab first image (no motion)
    motion_detect = motion_detector.MotionDetector(motion_min_area=args.minarea, screenwidth=args.screenwidth,
                                                   screenheight=args.screenheight, flip_camera=args.flipcamera,
                                                   iso=args.iso)  # init class
    output.message('Done with camera init... setting up classes.')
    bird_tweeter = tweeter.TweeterClass()  # init tweeter2 class twitter handler
    chores = dailychores.DailyChores(bird_tweeter, birdpop, cityweather, output_class=output)
    # init detection and classifier object
    birds = label_image.DetectClassify(homedir=args.homedir, classifier_labels=args.labels,
                                       classifier_model=args.classifier,
                                       classifier_thresholds=args.thresholds,
                                       detect_object_min_confidence=args.bird_confidence,
                                       classify_object_min_confidence=args.species_confidence,
                                       screenheight=args.screenheight, screenwidth=args.screenwidth,
                                       color_chg=args.color_chg,
                                       contrast_chg=args.contrast_chg, sharpness_chg=args.sharpness_chg,
                                       brightness_chg=args.brightness_chg, min_img_percent=args.minimgperc,
                                       target_object='bird',
                                       output_function=output.message, verbose=args.verbose)
    output.message(f'Using label file: {birds.labels}')
    output.message(f'Using threshold file: {birds.thresholds}')
    output.message(f'Using classifier file: {birds.classifier_file}')
    output.message('Starting while loop until sun set..... ')
    # loop while the sun is up, look for motion, detect birds, determine species
    while cityweather.sunrise.time() < datetime.now().time() < cityweather.sunset.time():
        chores.hourly_and_daily()  # initial weather reporting.  Should happen once between sun rise and +30 min.
        motion_detect.detect()
        if motion_detect.motion:
            motioncnt += 1

        if motion_detect.motion and birds.detect(img=motion_detect.img) and \
                cityweather.is_dawn() is False and cityweather.is_dusk() is False:  # daytime with motion and birds
            motioncnt = 0  # reset motion count between detected birds
            birds.set_colors()  # set new colors for this series of bounding boxes
            event_count += 1
            img_filename = os.getcwd() + '/assets/' + str(event_count % 10) + '.jpg'
            # output.message(message=f'Saw motion #{event_count} at {datetime.now().strftime("%I:%M:%S %P")}',
            #                msg_type='motion'
            #                event_num=event_count, image_name='')
            first_img_jpg = birds.img  # keep first shot for animation and web

            # classify, grab labels, output census, send to web and terminal,
            # enhance the shot, and add boxes, grab next set of gifs, build animation, tweet
            if birds.classify(img=first_img_jpg) >= args.species_confidence:  # found a bird we can classify
                first_rects, first_label, first_conf = birds.get_obj_data()  # grab data from this bird
                max_index = birds.classified_confidences.index(max(birds.classified_confidences))
                output.message(message=f'Possible sighting of a {birds.classified_labels[max_index]} '
                                       f'{birds.classified_confidences[max_index] * 100:.1f}% at '
                                       f'{datetime.now().strftime("%I:%M:%S %P")}', event_num=event_count,
                               msg_type='possible', image_name=img_filename, flush=True)  # label and confidence 2stream
                # check for need of final image adjustments, *** won't hit this code with twilight in while loop above
                first_img_jpg = first_img_jpg if args.brightness_chg == 0 \
                    or cityweather.isclear or cityweather.is_twilight() \
                    else image_proc.enhance_brightness(img=first_img_jpg, factor=args.brightness_chg)
                first_img_jpg_no_label = first_img_jpg.copy()
                # create animation: unlabeled first image is passed to gif function, bare copy is annotated later
                gif, gif_filename, animated, best_label, best_confidence, frames_with_birds = \
                    build_bird_animated_gif(args, motion_detect, birds, cityweather, first_img_jpg)

                # annotate bare image copy, use either best gif label or org data
                best_first_label = convert_to_list(best_label if best_label != '' else first_label)
                best_first_conf = convert_to_list(best_confidence if best_confidence > 0 else first_conf)
                bird_first_time_seen = birdpop.visitors(best_first_label, datetime.now())  # increment species count
                birds.set_ojb_data(classified_rects=first_rects, classified_labels=best_first_label,
                                   classified_confidences=best_first_conf)  # set to first bird
                first_img_jpg = birds.add_boxes_and_labels(img=first_img_jpg_no_label, use_last_known=False)
                first_img_jpg.save(img_filename)

                # process tweets, jpg if not min number of frame, gif otherwise
                waittime = birdpop.report_single_census_count(best_label) * args.tweetdelay / 10  # wait X min * N bird
                waittime = args.tweetdelay if waittime >= args.tweetdelay else waittime
                common_name, _, _ = parse_species(best_label)  # pull out common name from full species and sex
                if (datetime.now() - last_tweet).total_seconds() >= waittime or bird_first_time_seen or \
                        common_name in favorite_birds:
                    if animated:
                        output.message(message=f'Spotted {best_label} {best_confidence * 100:.1f}% '
                                               f'at {datetime.now().strftime("%I:%M:%S %P")}', event_num=event_count,
                                       msg_type='spotted', image_name=gif_filename, flush=True)
                        if bird_tweeter.post_image_from_file(tweet_text(best_label, best_confidence), gif_filename):
                            last_tweet = datetime.now()  # update last tweet time if successful gif posting, ignore fail
                    # elif best_confidence >= args.species_confidence:  # not animated, post jpg if high enough conf
                    #     tweet_jpg_text = tweet_text(best_first_label, best_first_conf)
                    #     output.message(message=f'Spotted {best_label} {best_confidence * 100:.1f}% '
                    #                            f'at {datetime.now().strftime("%I:%M:%S %P")}', event_num=event_count,
                    #                    msg_type='spotted',
                    #                    image_name=img_filename, flush=True)
                    #     if bird_tweeter.post_image_from_file(message=f'Sighted: {tweet_jpg_text}',
                    #                                          file_name=img_filename):
                    #         last_tweet = datetime.now()  # update last tweet time if successful, ignore fail
                    else:
                        output.message(message=f'Uncertain about a {best_label} {best_confidence * 100:.1f}% '
                                               f' with {frames_with_birds} frames with birds '
                                               f'at {datetime.now().strftime("%I:%M:%S %P")}',
                                               msg_type='inconclusive', event_num=event_count)

    output.end_stream()  # ending process for evening, print blank line and shut down
    motion_detect.stop()
    if args.verbose:
        chores.hourly_and_daily(report_pop=True)
        chores.end_report()  # post a report on run time of the process
    return  # to main and end process


def convert_to_list(input_str_list):
    return input_str_list if isinstance(input_str_list, list) else [input_str_list]


# should be passing default dictionary
# keeps track of count in label_dict (census) and confidences totals in conf_dict
# weighted dict is confidence total * (count-1) to weight away from low # of occurrences
def build_dict(label_dict, input_labels_list, conf_dict, input_confidences_list, weighted_dict):
    labels_list = convert_to_list(input_labels_list)
    confidences_list = convert_to_list(input_confidences_list)
    for ii, label in enumerate(labels_list):
        label_dict[label] += 1
        conf_dict[label] += confidences_list[ii]
        weighted_dict[label] = (label_dict[label] - 1) * conf_dict[label]
    return label_dict, conf_dict, weighted_dict


# loop thru keys and remove census entries with 1 or zero observations
def remove_single_observations(census_dict, conf_dict):
    key_list = []
    [key_list.append(key) if census_dict[key] <= 1 else '' for key in census_dict]
    [census_dict.pop(key) for key in key_list]
    [conf_dict.pop(key) for key in key_list]
    return census_dict, conf_dict


def build_bird_animated_gif(args, motion_detect, birds, cityweather, first_img_jpg):
    # grab a stream of pictures, add first pic from above, and build animated gif
    # return gif, filename, animated boolean, and best label as the max of all confidences
    gif_filename, best_label, best_confidence, labeled_frames = '', '', 0, []
    animated = False  # set to true if min # of frames captured with birds
    gif = first_img_jpg  # set a default if animated = False
    last_good_frame = 0  # find last frame that has a bird, index zero is good based on first image
    frames_with_birds = 1  # count of frames with birds, set to 1 for first img
    census_dict = defaultdict(default_value)  # track all results and pick best confidence
    confidence_dict = defaultdict(default_value)  # track all results and pick best confidence
    weighted_dict = defaultdict(default_value)  # conf * (count -1) to put emphasis on # of observations
    census_dict, confidence_dict, weighted_dict = build_dict(census_dict, birds.classified_labels, confidence_dict,
                                                             birds.classified_confidences, weighted_dict)
    frames = motion_detect.capture_stream()  # capture a list of images
    first_img_jpg = birds.add_boxes_and_labels(img=first_img_jpg)
    labeled_frames.insert(0, image_proc.convert_image(img=first_img_jpg, target='gif'))  # isrt 1st img
    for i, frame in enumerate(frames):
        frame = image_proc.enhance_brightness(img=frame, factor=args.brightness_chg)
        if birds.detect(img=frame):  # find bird object in frame and set rectangles containing object
            if birds.classify(img=frame, use_confidence_threshold=False) > 0:   # classify at rectangle & chk confidence
                frames_with_birds += 1
                last_good_frame = i + 1  # found a bird, add one to last good frame to account for insert of 1st image
            census_dict, confidence_dict, weighted_dict = build_dict(census_dict, birds.classified_labels,
                                                                     confidence_dict, birds.classified_confidences,
                                                                     weighted_dict)
        # frame = frame if args.brightness_chg == 0 \
        #     or cityweather.isclear or cityweather.is_twilight() \
        #     else image_proc.enhance_brightness(img=frame, factor=args.brightness_chg)  # increase bright
        labeled_frames.append(birds.add_boxes_and_labels(img=frame, use_last_known=True))  # use last label if unknown
    if frames_with_birds >= (args.minanimatedframes - 1):  # if bird is in min number of frames build gif
        gif, gif_filename = image_proc.save_gif(frames=labeled_frames[0:last_good_frame])  # framerate=motion_detect.FPS
        animated = True
    best_confidence = confidence_dict[max(confidence_dict, key=confidence_dict.get)] / \
        census_dict[max(confidence_dict, key=confidence_dict.get)]  # sum conf/bird cnt
    best_label = max(confidence_dict, key=confidence_dict.get)
    best_weighted_label = max(weighted_dict, key=weighted_dict.get)
    if best_label != best_weighted_label:
        print('--- Best label, confidence, and weight', best_label, best_confidence, best_weighted_label)
    return gif, gif_filename, animated, best_label, best_confidence, frames_with_birds


def tweet_text(label, confidence):
    # sample url https://www.allaboutbirds.org/guide/Northern_Rough-winged_Swallow/overview
    try:
        label = str(label[0]) if isinstance(label, list) else str(label)  # handle list or individual string
        confidence = float(confidence[0]) if isinstance(confidence, list) else float(confidence)  # list or float
        # sname = str(label)
        cname, sname, sex = parse_species(str(label))
        # sname = sname[sname.find(' ') + 1:] if sname.find(' ') >= 0 else sname  # remove index number
        # sex = sname[sname.find('[') + 1: sname.find(']')] if sname.find('[') >= 0 else ''  # retrieve sex
        # sname = sname[0: sname.find('[') - 1] if sname.find('[') >= 0 else sname  # remove sex
        # cname = sname[sname.find('(') + 1: sname.find(')')] if sname.find('(') >= 0 else sname  # retrieve common name
        hypername = cname.replace(' ', '_')
        hyperlink = f'https://www.allaboutbirds.org/guide/{hypername}/overview'
        tweet_label = f'{sex} {cname} {confidence * 100:.1f}% {hyperlink}'
    except Exception as e:
        tweet_label = ''
        print(e)
    return tweet_label


def parse_species(name):
    cname, sname, sex = '', '', ''
    try:
        sname = str(name)
        sname = sname[sname.find(' ') + 1:] if sname.find(' ') >= 0 else sname  # remove index number
        sex = sname[sname.find('[') + 1: sname.find(']')] if sname.find('[') >= 0 else ''  # retrieve sex
        sname = sname[0: sname.find('[') - 1] if sname.find('[') >= 0 else sname  # remove sex
        cname = sname[sname.find('(') + 1: sname.find(')')] if sname.find('(') >= 0 else sname  # retrieve common name
    except Exception as e:
        print(e)
    return cname, sname, sex


# def best_confidence_and_label(census_dict, confidence_dict):
#     best_confidence, best_confidence_label, best_census, best_census_label = 0, '', 0, ''
#     try:
#         census_dict, confidence_dict = remove_single_observations(census_dict, confidence_dict)  # multishots results
#         best_confidence = confidence_dict[max(confidence_dict, key=confidence_dict.get)] / \
#             census_dict[max(confidence_dict, key=confidence_dict.get)]  # sum conf/bird cnt
#         best_confidence_label = max(confidence_dict, key=confidence_dict.get)
#         best_census = census_dict[max(census_dict, key=census_dict.get)]
#         best_census_label = max(census_dict, key=census_dict.get)
#     except Exception as e:
#         print(e)
#     # print('best confidence:', best_confidence, best_confidence_label)
#     # print('best census:', best_census, best_census_label)
#     # if best_confidence_label != best_census_label:
#     #  what to do here?  Which one is better?  High count or high confidence?
#     return best_confidence, best_confidence_label, best_census, best_census_label


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    # camera settings
    ap.add_argument("-fc", "--flipcamera", type=bool, default=False, help="flip camera image")
    ap.add_argument("-sw", "--screenwidth", type=int, default=640, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=480, help="max screen height")

    ap.add_argument("-gf", "--minanimatedframes", type=int, default=8, help="minimum number of frames with a bird")
    ap.add_argument("-bb", "--broadcast", type=bool, default=False, help="stream images and text")
    ap.add_argument("-v", "--verbose", type=bool, default=True, help="To tweet extra stuff or not")
    ap.add_argument("-td", "--tweetdelay", type=int, default=1800,
                    help="Wait time between tweets is N species seen * delay/10 with not to exceed max of tweet delay")

    # motion and image processing settings, note adjustments are used as both a detector second prediction and a final
    # adjustment to the output images.
    ap.add_argument("-is", "--iso", type=int, default=800, help="iso camera sensitivity. higher requires less light")
    ap.add_argument("-b", "--brightness_chg", type=int, default=1.0, help="brightness boost")  # 1 no chg,< 1 -, > 1 +
    ap.add_argument("-c", "--contrast_chg", type=float, default=1.0, help="contrast boost")  # 1 no chg,< 1 -, > 1 +
    ap.add_argument("-cl", "--color_chg", type=float, default=1.0, help="color boost")  # 1 no chg,< 1 -, > 1 +
    ap.add_argument("-sp", "--sharpness_chg", type=float, default=1.0, help="sharpeness")  # 1 no chg,< 1 -, > 1 +

    # prediction defaults
    ap.add_argument("-sc", "--species_confidence", type=float, default=.960, help="species confidence threshold")
    ap.add_argument("-bc", "--bird_confidence", type=float, default=.6, help="bird confidence threshold")
    ap.add_argument("-ma", "--minarea", type=float, default=4.0, help="motion area threshold, lower req more")
    ap.add_argument("-ms", "--minimgperc", type=float, default=10.0, help="ignore objects that are less then % of img")

    ap.add_argument("-hd", "--homedir", type=str, default='/home/pi/PycharmProjects/birdclassifier/',
                    help="home directory for files")
    ap.add_argument("-la", "--labels", type=str, default='coral.ai.inat_bird_labels.txt',
                    help="name of file to use for species labels and thresholds")
    ap.add_argument("-tr", "--thresholds", type=str, default='coral.ai.inat_bird_threshold.csv',
                    help="name of file to use for species labels and thresholds")
    ap.add_argument("-cm", "--classifier", type=str, default='coral.ai.mobilenet_v2_1.0_224_inat_bird_quant.tflite',
                    help="model name for species classifier")
    # ap.add_argument("-la", "--labels", type=str, default='class_labels2022_07v1.txt',
    #                 help="name of file to use for species labels and thresholds")
    # ap.add_argument("-tr", "--thresholds", type=str, default='class_labels2022_07v1.csv',
    #                 help="name of file to use for species labels and thresholds")
    # ap.add_argument("-cm", "--classifier", type=str, default='mobilenet_tweeters2022_07v1.tflite',
    #                 help="model name for species classifier")

    ap.add_argument("-ct", "--city", type=str, default='Madison,WI,USA',
                    help="name of city weather station uses OWM web service.  See their site for city options")

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
