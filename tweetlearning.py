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
import pandas as pd
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
        text = str(dm[1]).lower()
        try:
            if text[0:5] == 'never':  # find species and mark as never, -1
                species_index = int(text[6: len(text)])
                species_thresholds.iloc[species_index, 1] = -1  # set to never
                print(f'species set to never: {species_thresholds.species[species_index]}')
            elif text[0:3] == 'inc':  # increase threshold for species by 10
                species_index = int(text[4: len(text)])
                species_thresholds.iloc[species_index,1] += 10  # increase threshold
                print(f'species threshold inc: {species_thresholds.species[species_index]}')
            else:
                print(f'direct message unknown request: {dm[1]}')
        except:
            print(f'direct message was not processed: {dm[1]}')
    return


# testing routine
def main(args):
    species_thresholds = pd.read_csv(args["species_thresholds"], delimiter=',', names=('species', 'threshold'))

    tweetersp = tweeter.Tweeter_Class()
    tweetersp.init(api_key, api_secret_key, access_token, access_token_secret)
    direct_messages = tweetersp.get_direct_messages()
    print('messages:')
    print(direct_messages)
    process_tweets(args, direct_messages, species_thresholds)  # parse tweet and modify thresholds
    tweetersp.destroy_direct_messages(direct_messages)  # destroy direct messages so they are not processed x2

    species_thresholds.to_csv(args['species_thresholds'], header=None, index=False)  # write out test file
    return


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-ts', '--species_thresholds',
                    default='c:/Users/maastricht/PycharmProjects/pyface2/coral.ai.inat_bird_threshold.csv',
                    help='name of file containing thresholds by label')
    ap.add_argument('-to', '--species_thresholds_out',
                    default='c:/Users/maastricht/PycharmProjects/pyface2/coral.ai.inat_bird_threshold_out.csv',
                    help='name of file to write out')

    logging.basicConfig(filename='birdclass.log', format='%(asctime)s - %(message)s', level=logging.DEBUG)
    arguments = vars(ap.parse_args())
    main(arguments)
