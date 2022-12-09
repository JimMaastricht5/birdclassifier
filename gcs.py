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


if __name__ == "__main__":
    jpg = Image.open('/home/pi/birdclass/0.jpg')
    web_storage = Storage()
    web_storage.send_file(blob_name='test0.jpg', blob_filename='/home/pi/birdclass/0.jpg')
