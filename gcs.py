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
        blob = self.bucket.blob(name)  # object name in bucket
        blob.upload_from_filename(file_loc_name)  # full qualified file location on disk

    def get_file(self, blob_name):
        blob = self.bucket.blob(blob_name)
        with blob.open("rb") as f:
            blob=f.read()
            p_image = Image.open(BytesIO(bytes(blob))) # Convert bytes to pil image
        return p_image

    def get_list(self, prefix=''):
        # use prefix= to get folder 'abc/myfolder'
        blob_name_list = []
        for blob in self.storage_client.list_blobs(self.bucket_name, prefix=prefix):
            if blob.name.find('.jpg') != -1 or blob.name.find('gif') != -1:  # append images to list
                blob_name_list.append(blob.name)
        return blob_name_list

    def get_all_files(self, blob_name_list):
        p_images = []
        for blob_name in blob_name_list:
            p_images.append(self.get_file(blob_name))
        return p_images


if __name__ == "__main__":
    # test send
    #jpg = Image.open('/home/pi/birdclass/0.jpg')
    # web_storage.send_file(blob_name='test0.jpg', blob_filename='/home/pi/birdclass/0.jpg')

    # get list test
    web_storage = Storage()
    blob_name_list = web_storage.get_list()

    # test retrieval in mem
    p_image = web_storage.get_file(blob_name_list[1])
    p_image.show()

    # test retrieval of all images
    # this takes a long time and will likely cost too much
    # for p_image in web_storage.get_all_files(blob_name_list):
    #     pass # just show the last one
    # p_image.show()

    # save file if needed
    # p_image.save("c:/home/pi/getblob.jpg")
