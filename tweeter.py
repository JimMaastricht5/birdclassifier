# module to tweet picture
# auth.py must be located in project; protect this file as it contains keys
from twython import Twython
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
def post_image(twitter, message, image):
    response = twitter.upload_media(media=image)
    twitter.update_status(status=message, media_ids=[response['media_id']])


# test code
def main_test():
    twitter = init_tweeter_connection(api_key, api_secret_key, access_token, access_token_secret)

    # test code to tweet a message
    message = 'Python status'
    post_status(message)
    print('tweeted: %s' % message)

    # test code to tweet a picture
    message = 'Python image test'
    image = open('cardinal.jpg', 'rb')
    post_image(message, image)
    print('tweeted: %s' % message)


if __name__ == "__main__":
    main_test()

