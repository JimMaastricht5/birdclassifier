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
# App to detect motion on a bird feeder, classify the object (is it a bird?),
# determine the species, record the appearance, send the image to twitter and the cloud for
# display. Web site at https://jimmaastricht5-tweetersp-1-main-veglvm.streamlit.app/
#
# uses tflite prebuilt google model for object detection and tensorflow lite model for bird classification
# Other models built with tensor flow  using tools from tensor in colab notebook
# https://colab.research.google.com/drive/1taZ9JincTaZuZh_JmBSC4pAbSQavxbq5#scrollTo=D3i_6WSXjUhk
# see read me for auth.py format requirements: https://github.com/JimMaastricht5/birdclassifier
# built by JimMaastricht5@gmail.com
import image_proc
import static_functions
from typing import Union
import det_class_label  # code to init tensor flow model and classify bird type, bird object
import motion_detector  # motion detector helper functions
import tweeter  # twitter helper functions
import population  # population census object, tracks species total seen and last time
import dailychores  # handle tasks that occur once per day or per hour
import weather
import output_stream
import argparse  # argument parser
import configparser  # read args from ini file
from datetime import datetime
import os
import uuid
import gcs  # save objects to website for viewing
import animate_gif  # animate gif class


# default dictionary returns a tuple of zero confidence and zero bird count
def default_value():
    # default dictionary returns a zero if asked for a key that doesn't exist
    return 0


def bird_detector(args) -> None:
    """
    main function for the bird feeder detector, takes a list of arguments from the command line or a file
    :param args: parsed arguments from arg parser list below
        "-cf", "--config_file", type=str, help='Config file'
        "-ol", "--offline", type=bool, default=False, help='Operate offline, do not transmit to cloud'
        "-db", "--debug", type=bool, default=False, help="debug flag"

        # camera settings
        "-fc", "--flipcamera", type=bool, default=False, help="flip camera image"
        "-sw", "--screenwidth", type=int, default=640, help="max screen width"
        "-sh", "--screenheight", type=int, default=480, help="max screen height"

        # general app settings
        "-gf", "--minanimatedframes", type=int, default=10, help="minimum number of frames with a bird"
        "-bb", "--broadcast", type=bool, default=False, help="stream images and text"
        "-v", "--verbose", type=bool, default=True, help="To tweet extra stuff or not"
        "-td", "--tweetdelay", type=int, default=1800
            help="Wait time between tweets is N species seen * delay/10 with not to exceed max of tweet delay"

        # motion and image processing settings,
        # note adjustments are used as both a detector second prediction and a final
        # adjustment to the output images.  # 1 no chg,< 1 -, > 1 +
        "-b", "--brightness_chg", type=int, default=1.2, help="brightness boost twilight"
        "-c", "--contrast_chg", type=float, default=1.0, help="contrast boost")  # 1 no chg,< 1 -, > 1 +
        "-cl", "--color_chg", type=float, default=1.0, help="color boost")  # 1 no chg,< 1 -, > 1 +
        "-sp", "--sharpness_chg", type=float, default=1.0, help="sharpness")  # 1 no chg,< 1 -, > 1 +

        # prediction defaults
        "-sc", "--species_confidence", type=float, default=.90, help="species confidence threshold"
        "-bc", "--bird_confidence", type=float, default=.6, help="bird confidence threshold"
        "-ma", "--minentropy", type=float, default=5.0, help="min change from first img to current to trigger motion"
        "-ms", "--minimgperc", type=float, default=10.0, help="ignore objects that are less than % of img"
        "-hd", "--homedir", type=str, default='/home/pi/birdclassifier/', help="home directory for files"
        "-la", "--labels", type=str, default='coral.ai.inat_bird_labels.txt', help="file for species labels "
        "-tr", "--thresholds", type=str, default='coral.ai.inat_bird_threshold.csv', help="file for species thresholds"
        "-cm", "--classifier", type=str, default='coral.ai.mobilenet_v2_1.0_224_inat_bird_quant.tflite',
            help="model name for species classifier"

        # feeder defaults
        "-ct", "--city", type=str, default='Madison,WI,USA', help="city name weather station uses OWM web service."
        '-fi', "--feeder_id", type=str, default=hex(uuid.getnode()), help='feeder id default MAC address'
        '-t', "--feeder_max_temp_c", type=int, default=86, help="Max operating temp for the feeder in C"
    :return: None
    """
    favorite_birds = ['Rose-breasted Grosbeak', 'Red-bellied Woodpecker']  # rare birds or just birds you want to see
    birdpop = population.Census()  # initialize species population census object
    output = output_stream.Controller(caller_id=args.city, debug=args.debug)  # handle terminal and web output
    output.start_stream()  # start streaming to terminal and web
    gcs_storage = gcs.Storage(offline=args.offline)
    motioncnt, event_count, gcs_img_filename, seed_check_gcs_filename = 0, 0, '', ''
    curr_day, curr_hr, last_tweet = (datetime.now().day, datetime.now().hour,
                                     datetime(2021, 1, 1, 0, 0, 0))

    # while loop below processes from sunrise to sunset.  The python program needs to be restarted daily
    # wait to enter that main while loop until sunrise
    cityweather = weather.CityWeather(city=args.city, units='Imperial', iscloudy=60, offline=False)  # get weather
    output.message(message=f'Now: {datetime.now()}.  \nSunrise: {cityweather.sunrise} Sunset: {cityweather.sunset}.',
                   msg_type='weather')
    cityweather.wait_until_midnight()  # if after sunset, wait here until after midnight
    cityweather.wait_until_sunrise()  # if before sun rise, wait here

    # initial video capture, screen size, and grab first image (no motion)
    motion_detect = motion_detector.MotionDetector(min_entropy=args.minentropy, screenwidth=args.screenwidth,
                                                   screenheight=args.screenheight, flip_camera=args.flipcamera,
                                                   first_img_name='first_img.jpg')
    # old code.... first_img_name = os.getcwd() + '/assets/' + 'first_img.jpg')  # init
    output.message('Done with camera init... setting up classes.')
    bird_tweeter = tweeter.TweeterClass(offline=args.offline)  # init tweeter2 class twitter handler
    chores = dailychores.DailyChores(bird_tweeter, birdpop, cityweather, output_class=output)
    # init detection and classifier object
    birds = det_class_label.DetectClassify(homedir=args.homedir, classifier_labels=args.labels,
                                           classifier_model=args.classifier,
                                           classifier_thresholds=args.thresholds,
                                           detect_object_min_confidence=args.bird_confidence,
                                           classify_object_min_confidence=args.species_confidence,
                                           screenheight=args.screenheight, screenwidth=args.screenwidth,
                                           color_chg=args.color_chg,
                                           contrast_chg=args.contrast_chg, sharpness_chg=args.sharpness_chg,
                                           brightness_chg=args.brightness_chg, min_img_percent=args.minimgperc,
                                           target_object=['bird'], debug=args.debug, output_class=output)
    output.message(f'Using label file: {birds.labels}')
    output.message(f'Using threshold file: {birds.thresholds}')
    output.message(f'Using classifier file: {birds.classifier_file}')
    output.message('Starting while loop until sun set..... ')

    bird_gif = animate_gif.BirdGif(motion_detector_cls=motion_detect, birds_cls=birds,
                                   gcs_storage_cls=gcs_storage, brightness_chg=args.brightness_chg,
                                   min_animated_frames=args.minanimatedframes,
                                   stash=args.offline)

    # loop while the sun is up, look for motion, detect birds, determine species
    while cityweather.sunrise.time() < datetime.now().time() < cityweather.sunset.time():
        chores.hourly_and_daily(filename=seed_check_gcs_filename)  # weather reporting, cpu checks, last img seed check
        motion_detect.detect()
        if motion_detect.motion:
            motioncnt += 1

        # check if day time, motion, bird, and not overexposed
        if motion_detect.motion and birds.detect(detect_img=motion_detect.img) and \
                cityweather.is_dawn() is False and cityweather.is_dusk() is False\
                and image_proc.is_sun_reflection_jpg(img=motion_detect.img, debug=args.debug) is False:
            motioncnt = 0  # reset motion count between detected birds
            birds.set_colors()  # set new colors for this series of bounding boxes
            event_count += 1  # increment event code for log and messages
            local_img_filename = os.getcwd() + '/assets/' + str(event_count % 10) + '.jpg'
            # gcs_img_filename = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}{str(event_count)}.jpg'  # redundant
            first_img_jpg = birds.img  # keep first shot for animation and web

            # classify, grab labels, output census, send to web and terminal,
            # enhance the shot, and add boxes, grab next set of gifs, build animation, tweet
            if birds.classify(class_img=first_img_jpg) >= args.species_confidence:  # found a bird we can classify
                first_rects, first_label, first_conf = birds.get_obj_data()  # grab data from this bird
                max_index = birds.classified_confidences.index(max(birds.classified_confidences))
                file_name = static_functions.common_name(birds.classified_labels[max_index]).replace(" ", "")
                gcs_img_filename = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}{str(event_count)}' \
                                   f'({file_name}).jpg'
                output.message(message=f'Possible sighting of a {birds.classified_labels[max_index]} '
                                       f'{birds.classified_confidences[max_index] * 100:.1f}% at '
                                       f'{datetime.now().strftime("%I:%M:%S %P")}', event_num=event_count,
                               msg_type='possible', image_name=gcs_img_filename, flush=True)  # 2 web stream
                # check for need of final image adjustments, *** won't hit this code with twilight in while loop above
                first_img_jpg = first_img_jpg if args.brightness_chg == 0 \
                    or cityweather.isclear or cityweather.is_twilight() \
                    else image_proc.enhance(img=first_img_jpg, brightness=args.brightness_chg)
                first_img_jpg_no_label = first_img_jpg.copy()
                # create animation: unlabeled first image is passed to gif function, bare copy is annotated later
                _gif = bird_gif.build_gif(event_count, first_img_jpg)

                # annotate bare image copy, use either best gif label or org data
                best_first_label = (
                    static_functions.convert_to_list(bird_gif.best_label if bird_gif.best_label != '' else first_label))
                best_first_conf = (
                    static_functions.convert_to_list(bird_gif.best_confidence if bird_gif.best_confidence > 0 else
                                                     first_conf))
                bird_first_time_seen = birdpop.visitors(best_first_label, datetime.now())  # increment species count
                birds.set_ojb_data(classified_rects=first_rects, classified_labels=best_first_label,
                                   classified_confidences=best_first_conf)  # set to first bird
                first_img_jpg = birds.add_boxes_and_labels(label_img=first_img_jpg_no_label, use_last_known=False)
                first_img_jpg.save(local_img_filename)
                gcs_storage.send_file(name=gcs_img_filename, file_loc_name=local_img_filename)
                seed_check_gcs_filename = gcs_img_filename  # reference to use for hourly seed check

                # process tweets, jpg if not min number of frame, gif otherwise.  wait X min * N bird before tweeting
                waittime = birdpop.report_single_census_count(bird_gif.best_label) * args.tweetdelay / 10
                waittime = args.tweetdelay if waittime >= args.tweetdelay else waittime
                if (datetime.now() - last_tweet).total_seconds() >= waittime or bird_first_time_seen or \
                        static_functions.common_name(bird_gif.best_label) in favorite_birds:
                    if bird_gif.animated:
                        output.message(message=f'Spotted {bird_gif.best_label} {bird_gif.best_confidence * 100:.1f}% '
                                               f'at {datetime.now().strftime("%I:%M:%S %P")}', event_num=event_count,
                                       msg_type='spotted', image_name=bird_gif.gcs_gif_filename, flush=True)
                        if bird_tweeter.post_image_from_file(tweet_text(bird_gif.best_label, bird_gif.best_confidence),
                                                             bird_gif.local_gif_filename):
                            last_tweet = datetime.now()  # update last tweet time if successful gif posting, ignore fail
                    else:
                        output.message(message=f'Uncertain about a {bird_gif.best_label} '
                                               f'{bird_gif.best_confidence * 100:.1f}% '
                                               f' with {bird_gif.frames_with_birds} frames with birds '
                                               f'at {datetime.now().strftime("%I:%M:%S %P")}',
                                               msg_type='inconclusive', event_num=event_count)

    output.end_stream()  # ending process for evening, print blank line and shut down
    motion_detect.stop()
    if args.verbose:
        chores.hourly_and_daily(report_pop=True)
        chores.end_report()  # post a report on run time of the process
    return  # to main and end process


# def convert_to_list(input_str_list):
#     """
#     :param input_str_list:
#     :return:
#     """
#     return input_str_list if isinstance(input_str_list, list) else [input_str_list]


def tweet_text(label: Union[list, str], confidence: Union[list, float]) -> str:
    """
    grab the best label and confidence, use that to generate a twitter text label
    :param label: list of labels or string contain label
    :param confidence: list of confidences or single float
    :return: tweet label as a string
    """
    # sample url https://www.allaboutbirds.org/guide/Northern_Rough-winged_Swallow/overview
    try:
        label = str(label[0]) if isinstance(label, list) else str(label)  # handle list or individual string
        confidence = float(confidence[0]) if isinstance(confidence, list) else float(confidence)  # list or float
        cname = static_functions.common_name(str(label))
        hypername = cname.replace(' ', '_')
        hyperlink = f'https://www.allaboutbirds.org/guide/{hypername}/overview'
        tweet_label = f'{cname} {confidence * 100:.1f}% {hyperlink}'
    except Exception as e:
        tweet_label = ''
        print(e)
    return tweet_label


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    # load settings from config file to allow for simple override
    ap = argparse.ArgumentParser()
    ap.add_argument("-cf", "--config_file", type=str, help='Config file')
    ap.add_argument("-ol", "--offline", type=bool, default=False,
                    help='Operate offline, do not transmit to cloud')
    ap.add_argument("-db", "--debug", type=bool, default=False, help="debug flag")

    # camera settings
    ap.add_argument("-fc", "--flipcamera", type=bool, default=False, help="flip camera image")
    ap.add_argument("-sw", "--screenwidth", type=int, default=480, help="max screen width")
    ap.add_argument("-sh", "--screenheight", type=int, default=640, help="max screen height")

    # general app settings
    ap.add_argument("-gf", "--minanimatedframes", type=int, default=10,
                    help="minimum number of frames with a bird")
    ap.add_argument("-bb", "--broadcast", type=bool, default=False, help="stream images and text")
    ap.add_argument("-v", "--verbose", type=bool, default=True, help="To tweet extra stuff or not")
    ap.add_argument("-td", "--tweetdelay", type=int, default=1800,
                    help="Wait time between tweets is N species seen * delay/10 with not to exceed max of tweet delay")

    # motion and image processing settings, note adjustments are used as both a detector second prediction and a final
    # adjustment to the output images.  # 1 no chg,< 1 -, > 1 +
    # ap.add_argument("-is", "--iso", type=int, default=800, help="iso camera sensitivity. higher requires less light")
    ap.add_argument("-b", "--brightness_chg", type=int, default=1.2, help="brightness boost twilight")
    ap.add_argument("-c", "--contrast_chg", type=float, default=1.0, help="contrast boost")
    ap.add_argument("-cl", "--color_chg", type=float, default=1.0, help="color boost")
    ap.add_argument("-sp", "--sharpness_chg", type=float, default=1.0, help="sharpness")

    # prediction defaults
    ap.add_argument("-sc", "--species_confidence", type=float, default=.90,
                    help="species confidence threshold")
    ap.add_argument("-bc", "--bird_confidence", type=float, default=.6, help="bird confidence threshold")
    ap.add_argument("-ma", "--minentropy", type=float, default=5.0,
                    help="min change from first img to current to trigger motion")
    ap.add_argument("-ms", "--minimgperc", type=float, default=10.0,
                    help="ignore objects that are less then % of img")
    ap.add_argument("-hd", "--homedir", type=str, default='/home/pi/birdclassifier/',
                    help="home directory for files")
    ap.add_argument("-la", "--labels", type=str, default='coral.ai.inat_bird_labels.txt',
                    help="name of file to use for species labels and thresholds")
    ap.add_argument("-tr", "--thresholds", type=str, default='USA_WI_coral.ai.inat_bird_threshold.csv',
                    help="name of file to use for species labels and thresholds")
    ap.add_argument("-cm", "--classifier", type=str,
                    default='coral.ai.mobilenet_v2_1.0_224_inat_bird_quant.tflite',
                    help="model name for species classifier")

    # feeder defaults
    ap.add_argument("-ct", "--city", type=str, default='Madison,WI,USA',
                    help="name of city weather station uses OWM web service.  See their site for city options")
    ap.add_argument('-fi', "--feeder_id", type=str, default=hex(uuid.getnode()),
                    help='feeder id default MAC address')
    ap.add_argument('-t', "--feeder_max_temp_c", type=int, default=86,
                    help="Max operating temp for the feeder in C")
    arguments = ap.parse_args()
    if arguments.config_file:
        config = configparser.ConfigParser()
        config.read(arguments.config_file)
        defaults = {}
        defaults.update(dict(config.items()))
        ap.set_defaults(**defaults)
        arguments = ap.parse_args()  # Overwrite arguments with config file

    bird_detector(arguments)
