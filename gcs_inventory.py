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
import gcs
import static_functions
import datetime as dt
import pandas as pd

def get_archived_jpg_images(save_file_name: str='archive-jpg-list.csv'):
    gcs_store = gcs.Storage(bucket_name='archive_jpg_from_birdclassifier')
    image_list = gcs_store.get_img_list()
    image_dict = []
    image_date_time_str = ''
    print(f'Found {len(image_list)} images')
    for ii, image_name in enumerate(image_list):
        try:
            common_name = static_functions.common_name(image_name)
            image_date_time_str = image_name[0:18]  # date time string is in first 19 characters of the files name
            img_date_time = dt.datetime.strptime(image_date_time_str, '%Y-%m-%d-%H-%M-%S')
            image_dict.append({'Number': ii, 'Name': common_name, 'Year': img_date_time.year, 'Month': img_date_time.month,
                              'Day': img_date_time.day, 'Hour': img_date_time.hour, 'Seconds': img_date_time.second,
                              'Image Name': image_name})
        except ValueError as e:
            print(f'Value Error {e}')
            print(f'image date time string {image_date_time_str}')
            print(f'Image file name: {image_name}')

    df = pd.DataFrame(image_dict)
    df.to_csv(save_file_name, index=False)
    return df


if __name__ == "__main__":
    df_file_name = 'archive-jpg-list.csv'
    # df = get_archived_jpg_images('archive-jpg-list.csv')
    df = pd.read_csv(df_file_name)
    name_counts = df['Name'].value_counts()
    print(df.columns)
    # print(f'Name Counts: {name_counts}')
    print(f'Possible False Positives: {name_counts[name_counts <= 150]}')
    print(f'Remaining Species: {name_counts[name_counts > 150]}')
    print(len(name_counts))
    # print(df.head(5).to_string())

