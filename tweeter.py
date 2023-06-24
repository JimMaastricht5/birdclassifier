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
import requests
import base64
import json
import tweepy
import numpy as np
from datetime import datetime
from auth import (
    api_key,
    api_secret_key,
    access_token,
    access_token_secret,
    bearer_token,
    client_id,
    client_secret
)


class TweeterClass:

    def __init__(self, tweetmax_per_hour=15):
        self.twitter = self.init(api_key, api_secret_key, access_token, access_token_secret)
        self.curr_day = datetime.now().day
        self.curr_hr = datetime.now().hour
        self.tweetcnt = 0
        self.tweetmax_per_hour = tweetmax_per_hour
        self.tweeted = False

    # initialize twitter connection and login
    def init(self, consumer_key, consumer_secret, access_token, access_token_secret):
        # auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        # auth.set_access_token(access_token, access_token_secret)
        # api = tweepy.API(auth)  # api v1
        api = tweepy.Client(bearer_token=bearer_token, consumer_key=consumer_key, consumer_secret=consumer_secret,
                            access_token=access_token, access_token_secret=access_token_secret)  # api v2
        return api

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
            try:
                # self.twitter.update_status(status=message)
                self.twitter.create_tweet(text=message)
                print(message)
                self.tweeted = True
            except Exception as e:
                print(e)
                self.tweeted = False
        else:
            self.tweeted = False
        return

    def post_image_url(self, message, url):
        self.check_hour()
        if self.tweetcnt < self.tweetmax_per_hour:
            try:
                # media = self.twitter.media_upload(filename=file_name)
                # self.twitter.update_status(status=message, media_ids=[media.media_id])
                self.twitter.create_tweet(text=url + ' ' + message)
                print(message)
                self.tweetcnt += 1
                self.tweeted = True
            except Exception as e:
                print(e)
                self.tweeted = False
        else:
            self.tweeted = False
        return self.tweeted

    # set status and add an image
    def post_image_from_file(self, message, file_name):
        # need to build new version
        return

    def post_image_from_file_v2(self, message, file_name):
        # Read the media file as binary data
        with open(file_name, 'rb') as file:
            media_data = file.read()

        # Base64 encode the media file
        base64_media = base64.b64encode(media_data)

        # Make the POST request to upload the media
        upload_url = 'https://upload.twitter.com/1.1/media/upload.json'
        headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
        }
        data = {
            'media_data': base64_media.decode('utf-8')
        }
        response = requests.post(upload_url, headers=headers, data=data)
        print(response)
        response_data = json.loads(response.text)

        # Get the media ID from the response
        media_id = response_data['media_id']

        # Create a tweet with the uploaded media
        tweet_text = message
        tweet_url = 'https://api.twitter.com/1.1/statuses/update.json'
        headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
        }
        data = {
            'status': tweet_text,
            'media_ids': media_id
        }
        response = requests.post(tweet_url, headers=headers, data=data)

        # Check the response status
        if response.status_code == 200:
            print('Tweet successfully posted.')
        else:
            print('Error posting tweet:', response.text)
        return

    # api v1 code
    # def post_image_from_file(self, message, file_name):
    #     self.check_hour()
    #     if self.tweetcnt < self.tweetmax_per_hour:
    #         try:
    #             media = self.twitter.media_upload(filename=file_name)
    #             self.twitter.update_status(status=message, media_ids=[media.media_id])
    #             self.tweetcnt += 1
    #             self.tweeted = True
    #         except Exception as e:
    #             print(e)
    #             self.tweeted = False
    #     else:
    #         self.tweeted = False
    #     return self.tweeted
    #
    # # set status and add an image
    # def post_image(self, message, img, file_name):
    #     self.check_hour()
    #     if self.tweetcnt < self.tweetmax_per_hour:
    #         try:
    #             # img.save(file_name)
    #             # response = self.twitter.upload_media(media=img)  # doesn't work!
    #             # media = self.twitter.media_upload(filename=file_name, file=img)  # doesn't work
    #             media = self.twitter.media_upload(filename=file_name)
    #             self.twitter.update_status(status=message, media_ids=[media.media_id])
    #             self.tweetcnt += 1
    #             self.tweeted = True
    #         except Exception as e:
    #             print(e)
    #             self.tweeted = False
    #     else:
    #         self.tweeted = False
    #     return self.tweeted

    # get direct messages, returns numpy array with x, 2 shape
    # def get_direct_messages(self):
    #     dm_array = []
    #     direct_messages = self.twitter.get_direct_messages()  # returns json
    #     for dm in direct_messages['events']:  # unpack json and build list
    #         dm_array.append((dm['id'], dm['message_create']['message_data']['text']))  # insert row of 2 columns
    #     return np.array(dm_array)

    # destroy all direct messages, takes numpy array with id and text
    # def destroy_direct_messages(self, direct_messages):
    #     for dmid, dmtext in direct_messages:
    #         self.twitter.destroy_direct_message(id=dmid)
    #     return


# test code
def main_test():
    tweeter_obj = TweeterClass()

    # test code to tweet a message
    message = 'Python status'
    tweeter_obj.post_status(message)
    print('tweeted: %s' % message)

    # test code to tweet a picture
    message = 'Python image test'
    url = 'https://storage.googleapis.com/tweeterssp-web-site-contents/2023-05-19-08-04-37167(CommonGrackle).jpg'
    # twtimage = open('cardinal.jpg', 'rb')
    # tweeter_obj.post_image_url(message, url)
    tweeter_obj.post_image_from_file_v2(message='test', file_name='cardinal.jpg')
    print('tweeted: %s' % message)


if __name__ == "__main__":
    main_test()
