# MIT License
#
# 2024 Jim Maastricht
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
import pandas
import pandas as pd
from google.cloud import storage
from PIL import Image  # Pillow
from io import BytesIO  # built-in package
try: 
    from auth import (
        google_json_key
    )
except ModuleNotFoundError:
    google_json_key = None
    print('no module auth.py found with key google_json_key for gcs, assuming this is running from within GCP project')
    pass

class Storage:
    """
    Class makes reading and writing from a Google cloud storage (GCS) bucket easy....
    """
    def __init__(self, project: str = "birdvision", bucket_name: str = "tweeterssp-web-site-contents",
                 offline: bool = False) -> None:
        """
        :param project: the Google project that the buckets lives in
        :param bucket_name: the name of the bucket to read and write files
        :param offline: tells the class not to send objects to the web, used for testing or running with no wifi
        """
        # connect to temp bucket for public send
        self.project = project
        self.offline = offline
        # print(self.offline)
        if self.offline:
            self.storage_client = None
            self.bucket = None
            self.bucket_name = None
        else:
            if google_json_key is None:
                self.storage_client = storage.Client()  # from within GCP project
            else:
                self.storage_client = storage.Client.from_service_account_json(json_credentials_path=google_json_key)  # from feeder
            self.bucket = self.storage_client.bucket(bucket_name)
            self.bucket_name = bucket_name


    def send_file(self, name: str, file_loc_name: str) -> None:
        """
        send a file to google cloud storage
        :param name: name of the file to write into the cloud bucket
        :param file_loc_name: fully qualified location on disk to grab from
        :return:
        """
        if self.offline:
            return

        try:
            blob = self.bucket.blob(name)  # object name in bucket
            blob.upload_from_filename(file_loc_name)  # full qualified file location on disk
        except Exception as e:
            print(f'Error in GCS send file {e}')
        return

    def send_df(self, df: pandas.DataFrame, blob_name: str) -> None:
        """
        write a dataframe to gcs
        :param df: pandas data frame to write to cloud
        :param blob_name: name of the object to write to in GCS
        :return:
        """
        if self.offline:
            return

        try:
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(df.to_csv(), 'text/csv')
        except Exception as e:
            print(f'failure sending dataframe to GCS: {e}')
        return

    def get_img_file(self, blob_name: str) -> Image.Image:
        """
        Get an img from cloud storage
        :param blob_name: name of the blob to pull
        :return: image requested
        """
        blob = self.bucket.blob(blob_name)
        with blob.open("rb") as f:
            blob = f.read()
            gcs_image = Image.open(BytesIO(bytes(blob)))  # Convert bytes to pil image
        return gcs_image

    def get_img_list(self, prefix: str = '') -> list:
        """
        generate a list of all images in a GCS bucket, ignore other file types
        :param prefix: string specifying folder structure within the bucket to grab from
        :return:
        """
        # use prefix= to get folder 'abc/myfolder'
        blob_name_list = []
        for blob in self.storage_client.list_blobs(self.bucket_name, prefix=prefix):
            # print(blob.name)
            if blob.name.find('.jpg') != -1 or blob.name.find('gif') != -1:  # append images to list
                blob_name_list.append(blob.name)
        return blob_name_list

    def get_all_img_files(self, blob_name_list: list) -> list:
        """
        get a list of images from a bucket
        :param blob_name_list: list of blob names to retrieve
        :return: list of images
        """
        images = []
        for blob_name in blob_name_list:
            images.append(self.get_img_file(blob_name))
        return images

    def get_csv_list(self, prefix: str = '') -> list:
        """
        grab a list of csv files in a bucket
        :param prefix: specifies folder structure, use prefix= to get folder 'abc/myfolder'
        :return: list of csv files in bucket
        """
        blob_name_list = []
        for blob in self.storage_client.list_blobs(self.bucket_name, prefix=prefix):
            if blob.name.find('.csv') != -1:  # append csv to list
                blob_name_list.append(blob.name)
        return blob_name_list

    def get_df(self, blob_name: str, header: int=0, column_names: list=None, delimiter: str = ',') -> pandas.DataFrame:
        """
        load a dataframe from a blob name
        :param blob_name: string containing the name of the file to load
        :param header: int default is 0 meaning the df has a heading, send none to indicate no heading
        :param column_names: list containing columns names
        :return: populated dataframe
        """
        blob = self.bucket.blob(blob_name)
        blob_bytes = blob.download_as_bytes()
        if header == 0:
            df = pd.read_csv(BytesIO(blob_bytes), header=header, delimiter=delimiter)
        else:
            df=pd.read_csv(BytesIO(blob_bytes), header=header, names=column_names, delimiter=delimiter)
        return df


# testing code for this module
if __name__ == "__main__":
    # get a list of files from archive for rasp pi perf test
    web_storage = Storage(bucket_name='archive_jpg_from_birdclassifier')
    df = pd.read_csv('archive-jpg-list-2025-sample.csv')
    blob_name_list = df['Image Name'].tolist()
    for blob_name in blob_name_list:
        print(f'Getting image name {blob_name}')
        p_image = web_storage.get_img_file(blob_name)
        print(f'Saving image name {blob_name}')
        p_image.save(f'/home/pi/batch_test/{blob_name}')

    # old testing code
    # web_storage = Storage()

    # test crawling sub directories with get image list
    # nabirds_storage = Storage(bucket_name='nabirds_filtered')
    # nabirds_list = nabirds_storage.get_img_list(prefix='images/')
    # print(nabirds_list)


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
    # last_gif_name = ''
    # file_name_list = web_storage.get_img_list()
    # file_name_list.reverse()
    # for file_name in file_name_list:
    #     if file_name.find('.gif') != -1:
    #         last_gif_name = file_name
    #         break
    # print(file_name_list)
    # print(last_gif_name)
    #
    # # test retrieval in mem
    # p_image = web_storage.get_img_file(file_name_list[1])
    # p_image.show()

    # test retrieval of all images
    # this takes a long time and will likely cost too much
    # for p_image in web_storage.get_all_files(blob_name_list):
    #     pass # just show the last one
    # p_image.show()
    # p_image.save("c:/home/pi/getblob.jpg")



