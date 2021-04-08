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
# auth.py must be located in project; protect this file as it contains keys
# code by JimMaastricht5@gmail.com
from twython import Twython
import numpy as np
import cv2
from datetime import datetime
from auth import (
    api_key,
    api_secret_key,
    access_token,
    access_token_secret
)

class Tweeter_Class:

    def __init__(self):
        self.twitter = self.init(api_key, api_secret_key, access_token, access_token_secret)
        self.curr_day = datetime.now().day
        self.curr_hr = datetime.now().hour
        self.tweetcnt = 0
        self.tweetmax_per_hour = 25
        self.tweeted = False

    # initialize twitter connection and login
    def init(self, api_key, api_secret_key, access_token, access_token_secret):
        return Twython(api_key, api_secret_key, access_token, access_token_secret)

    # reset hourly tweet count if new hour
    def check_hour(self):
        if self.curr_hr != datetime.now().hour:
            self.curr_hr = datetime.now().hour
            self.tweetcnt = 0
        return

    # set status message
    def post_status(self, message):
        self.check_hour()
        if self.tweetcnt < self.tweetmax_per_hour:
            self.tweetcnt += 1
            self.twitter.update_status(status=message)
            self.tweeted = True
        else:
            self.tweeted = False
        return

    # set status and add an image
    def post_image(self, message, img):
        self.check_hour()
        cv2.imwrite("img.jpg", img)  # write out image for debugging and testing
        tw_img = open('img.jpg', 'rb')  # reload a image for twitter, correct var type
        if self.tweetcnt < self.tweetmax_per_hour:
            response = self.twitter.upload_media(media=tw_img)
            self.twitter.update_status(status=message, media_ids=[response['media_id']])
            self.tweetcnt += 1
            self.tweeted = True
        else:
            self.tweeted = False
        return

    # get direct messages, returns numpy array with x, 2 shape
    def get_direct_messages(self):
        dm_array = []
        direct_messages = self.twitter.get_direct_messages()  # returns json
        for dm in direct_messages['events']:  # unpack json and build list
            dm_array.append((dm['id'], dm['message_create']['message_data']['text']))  # insert row of 2 columns
        return np.array(dm_array)

    # destroy all direct messages, takes numpy array with id and text
    def destroy_direct_messages(self, direct_messages):
        for dmid, dmtext in direct_messages:
            self.twitter.destroy_direct_message(id=dmid)
        return


# test code
def main_test():
    tweeter_obj = Tweeter_Class()
    direct_messages = tweeter_obj.get_direct_messages()
    print(direct_messages)
    print(direct_messages.shape)
    tweeter_obj.destroy_direct_messages(direct_messages)

    # test code to tweet a message
    # message = 'Python status'
    # tweeter_obj.post_status(message)
    # print('tweeted: %s' % message)
    #
    # test code to tweet a picture
    # message = 'Python image test'
    # twtimage = open('cardinal.jpg', 'rb')
    # tweeter_obj.post_image(message, twtimage)
    # print('tweeted: %s' % message)


if __name__ == "__main__":
    main_test()
