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
from auth import (
    api_key,
    api_secret_key,
    access_token,
    access_token_secret
)


# initialize twitter connection and login
def init(api_key, api_secret_key, access_token, access_token_secret):
    return Twython(api_key, api_secret_key, access_token, access_token_secret)


# set status message
def post_status(twitter, message):
    return twitter.update_status(status=message)


# set status and add an image
def post_image(twitter, message, img):
    response = twitter.upload_media(media=img)
    twitter.update_status(status=message, media_ids=[response['media_id']])


# get direct messages, returns numpy array with x, 2 shape
def get_direct_messages(twitter):
    dm_array = []
    direct_messages = twitter.get_direct_messages()  # returns json
    for dm in direct_messages['events']:  # unpack json and build list
        dm_array.append( (dm['id'], dm['message_create']['message_data']['text']) ) # insert row of 2 columns
    return np.array(dm_array)


# destroy all direct messages, takes numpy array with id and text
def destroy_direct_messages(twitter, direct_messages):
    for dmid, dmtext in direct_messages:
        twitter.destroy_direct_message(id=dmid)
    return


# test code
def main_test():
    twitter = init(api_key, api_secret_key, access_token, access_token_secret)

    direct_messages = get_direct_messages(twitter)
    print(direct_messages)
    print(direct_messages.shape)
    destroy_direct_messages(twitter, direct_messages)

    # test code to tweet a message
    # message = 'Python status'
    # post_status(twitter, message)
    # print('tweeted: %s' % message)
    #
    # test code to tweet a picture
    # message = 'Python image test'
    # twtimage = open('cardinal.jpg', 'rb')
    # post_image(twitter, message, twtimage)
    # print('tweeted: %s' % message)


if __name__ == "__main__":
    main_test()
