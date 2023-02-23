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
# module handles all file interactions with google cloud storage
# google keys must be in a python file called auth.py and that should never be included in git!
# send file and send df were hardened to protect the bird feeder app from crashing,
# that is unnecessary for the get functions since that is used by the web app and the page can be refreshed to
# reinvoke the app
import pandas as pd
from google.cloud import storage
from PIL import Image
from io import BytesIO
from auth import (
    google_json_key
)


class Storage:
    def __init__(self, project="birdvision", bucket_name="tweeterssp-web-site-contents"):
        # connect to temp bucket for public send
        self.project = project
        self.storage_client = storage.Client.from_service_account_json(json_credentials_path=google_json_key)
        self.bucket = self.storage_client.bucket(bucket_name)
        self.bucket_name = bucket_name

    def send_file(self, name, file_loc_name):
        try:
            blob = self.bucket.blob(name)  # object name in bucket
            blob.upload_from_filename(file_loc_name)  # full qualified file location on disk
        except Exception as e:
            print(f'Error in GCS send file {e}')
        return

    def send_df(self, df, blob_name):
        try:
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(df.to_csv(), 'text/csv')
        except Exception as e:
            print(f'failure sending dataframe to GCS: {e}')
        return

    def get_img_file(self, blob_name):
        blob = self.bucket.blob(blob_name)
        with blob.open("rb") as f:
            blob = f.read()
            gcs_image = Image.open(BytesIO(bytes(blob)))  # Convert bytes to pil image
        return gcs_image

    def get_img_list(self, prefix=''):
        # use prefix= to get folder 'abc/myfolder'
        blob_name_list = []
        for blob in self.storage_client.list_blobs(self.bucket_name, prefix=prefix):
            if blob.name.find('.jpg') != -1 or blob.name.find('gif') != -1:  # append images to list
                blob_name_list.append(blob.name)
        return blob_name_list

    def get_all_img_files(self, blob_name_list):
        images = []
        for blob_name in blob_name_list:
            images.append(self.get_img_file(blob_name))
        return images

    def get_csv_list(self, prefix=''):
        # use prefix= to get folder 'abc/myfolder'
        blob_name_list = []
        for blob in self.storage_client.list_blobs(self.bucket_name, prefix=prefix):
            if blob.name.find('.csv') != -1:  # append csv to list
                blob_name_list.append(blob.name)
        return blob_name_list

    def get_df(self, blob_name):
        blob = self.bucket.blob(blob_name)
        blob_bytes = blob.download_as_bytes()
        df = pd.read_csv(BytesIO(blob_bytes), header=0)
        return df


# testing code for this module
if __name__ == "__main__":
    web_storage = Storage()

    # test dataframe func
    # csv_list = web_storage.get_csv_list()
    # # print(csv_list)
    # for csv in csv_list:
    #     df = web_storage.get_df(csv)
    #
    # print(df.columns)
    # print(df.iloc[0])

    # test send
    # jpg = Image.open('/home/pi/birdclass/0.jpg')
    # web_storage.send_file(blob_name='test0.jpg', blob_filename='/home/pi/birdclass/0.jpg')

    # get list test
    last_gif_name = ''
    file_name_list = web_storage.get_img_list()
    file_name_list.reverse()
    for file_name in file_name_list:
        if file_name.find('.gif') != -1:
            last_gif_name = file_name
            break
    print(file_name_list)
    print(last_gif_name)

    # test retrieval in mem
    p_image = web_storage.get_img_file(file_name_list[1])
    p_image.show()

    # test retrieval of all images
    # this takes a long time and will likely cost too much
    # for p_image in web_storage.get_all_files(blob_name_list):
    #     pass # just show the last one
    # p_image.show()

    # save file if needed
    # p_image.save("c:/home/pi/getblob.jpg")
