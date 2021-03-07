# module to tweet picture
# auth.py must be located in project; protect this file as it contains keys
# code by JimMaastricht5@gmail.com
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
def post_image(twitter, message, img):
    response = twitter.upload_media(media=img)
    twitter.update_status(status=message, media_ids=[response['media_id']])


# get direct messages
def get_direct_messages(twitter):
    direct_messages = twitter.get_direct_messages()
    for dm in direct_messages['events']:
        id = dm['id']
        print(id, dm['message_create']['message_data']['text'])
        # twitter.destroy_direct_message(id)
        # break
    return


# test code
def main_test():
    twitter = init(api_key, api_secret_key, access_token, access_token_secret)

    get_direct_messages(twitter)
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
