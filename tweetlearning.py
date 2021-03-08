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
# module runs daily to check twitter for direct messages
# messages with "never" mean the species is not found in WI; # is the species key in the file, e.g., never 965
# messages with "inc" mean the threshold for this species may be set too low; # is the species, increments by 10
# process messages and adjust species threshold file coral.ai.inat_bird_threshold.csv
import tweeter  # twitter helper functions
import argparse  # argument parser
import numpy as np
from datetime import datetime
from auth import (
    api_key,
    api_secret_key,
    access_token,
    access_token_secret
)  # import twitter keys
import logging


# read direct messages and process accordingly, return boolean true/false if an adjustment was made to the file
def process_tweets(args, direct_messages, species_thresholds):
    for dm in direct_messages:
        text = dm[1].lower()
        print(text)
        try:
            if text.substring(0, 5) == 'never':  # find species and mark as never, -1
                species_index = int(text.substring(6, len(text)))
                species_thresholds[species_index][1] = -1  # set to never
                print(f'species set to never: {species_thresholds[species_index][0]}')
            elif text == 'inc':  # increase threshold for species by 10
                species_index = int(text.substring(4, len(text)))
                species_thresholds[species_index][1] += 10  # increase threshold
                print(f'species threshold inc: {species_thresholds[species_index][0]}')
            else:
                print(f'direct message unknown request: {dm[1]}')
        except:
            print(f'direct message was not procesed: {dm[1]}')
    # np.savetxt(args["species_thresholds"], delimiter=',', fmt='%s')  # write out file as strings comma delimit
    return


# testing routine
def main(args):
    species_thresholds = np.genfromtxt(args["species_thresholds"], delimiter=',')  # load species threshold file
    twitter = tweeter.init(api_key, api_secret_key, access_token, access_token_secret)
    direct_messages = tweeter.get_direct_messages(twitter)
    process_tweets(args, direct_messages, species_thresholds)
    return


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-ts', '--species_thresholds',
                    default='c:/Users/maastricht/PycharmProjects/pyface2/coral.ai.inat_bird_threshold.csv',
                    help='name of file containing thresholds by label')

    logging.basicConfig(filename='birdclass.log', format='%(asctime)s - %(message)s', level=logging.DEBUG)
    arguments = vars(ap.parse_args())
    main(arguments)