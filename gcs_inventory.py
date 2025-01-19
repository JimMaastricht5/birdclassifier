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
# reads the set of files in gcs storage for archived jpgs and creates a csv for further analysis
import pandas
import gcs
import static_functions
import datetime as dt
import pandas as pd
import re


def insert_spaces_before_capitals(text: str) -> str:
    """
    Inserts spaces before capital letters (excluding the first one) in a string.
    :param text: string with text to change
    :return: The modified string with spaces inserted, or the original string if no changes are needed.
                Returns an empty string if input is None or empty.
    """
    return ("" if not text else re.sub(r"(?<!^)([A-Z])", r" \1", text)).replace('- ', '')


def get_archived_jpg_images(save_file_name: str='archive-jpg-list.csv') -> pandas.DataFrame:
    """
    get list of web archived jpgs from gcs, put into df, and write csv to output
    :param save_file_name: string containing the file name
    :return: pandas.Dataframe
    """
    gcs_store = gcs.Storage(bucket_name='archive_jpg_from_birdclassifier')
    image_list = gcs_store.get_img_list()
    image_dict = []
    image_date_time_str = ''
    for ii, image_name in enumerate(image_list):
        try:
            common_name = insert_spaces_before_capitals(static_functions.common_name(image_name))
            image_date_time_str = image_name[0:18]  # date time string is in first 19 characters of the files name
            img_date_time = dt.datetime.strptime(image_date_time_str, '%Y-%m-%d-%H-%M-%S')
            image_dict.append({'Number': ii, 'Name': common_name, 'Year': img_date_time.year, 'Month': img_date_time.month,
                              'Day': img_date_time.day, 'Hour': img_date_time.hour, 'DateTime': img_date_time,
                               'Image Name': image_name})
        except ValueError as e:
            print(f'Value Error {e}')
            print(f'image date time string {image_date_time_str}')
            print(f'Image file name: {image_name}')
    df_img = pd.DataFrame(image_dict)
    df_img.to_csv(save_file_name, index=False)
    return df_img


if __name__ == "__main__":
    df_file_name = 'archive-jpg-list.csv'
    # _ = get_archived_jpg_images(df_file_name)  # .01 per 1000 operations so about $0.80 per run
    df_raw = pd.read_csv(df_file_name)
    df_raw['DateTime'] = pd.to_datetime(df_raw['DateTime'], errors='raise')
    print('Limiting list to 2024 only....')
    df = df_raw[df_raw['DateTime'].dt.year == 2024].copy() # .copy() avoids warnings about setting values on slice
    name_counts = df['Name'].value_counts()
    print(df.columns)
    print('')
    print(f'Starting date: {df['DateTime'].min()}')
    print(f'Ending date: {df['DateTime'].max()}')
    print(f'Number of Images: \t{df.shape[0]}\n')
    print(f'Possible False Positives: \n{name_counts[name_counts <= 150]}')
    print('')
    print(f'Remaining Species: \n{name_counts[name_counts > 150]}')

    # limit to 2024 for full year of analysis



