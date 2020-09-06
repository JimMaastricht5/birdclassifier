# module to tweet picture
# auth.py must be located in project; protect this file as it contains keys

from twython import Twython
from auth import (
    api_key,
    api_secret_key,
    access_token,
    access_token_secret
)

def init_tweeter_connection(api_key, api_secret_key, access_token, access_token_secret):
    return Twython(api_key, api_secret_key, access_token, access_token_secret)

def tweet_status(message):
    return twitter.update_status(status=message)

def tweet_image(message, image):
    response = twitter.upload_media(media=image)
    twitter.update_status(status=message, media_ids=[response['media_id']])


# test code
twitter = init_tweeter_connection(api_key, api_secret_key, access_token, access_token_secret)

# test code to tweet a message
message = 'Python image test2'
# tweet_status(message)
# print('tweeted: %s' % message)

# test code to tweet a picture
image = open('cardinal.jpg', 'rb')
tweet_image(message, image)


