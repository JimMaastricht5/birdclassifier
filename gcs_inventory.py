import gcs
import static_functions
import datetime as dt
import pandas as pd

# if __name__ == "__main__":
gcs_store = gcs.Storage(bucket_name='archive_jpg_from_birdclassifier')
image_list = gcs_store.get_img_list()
image_dict = []
print(f'Found {len(image_list)} images')
for ii, image_name in enumerate(image_list):
    common_name = static_functions.common_name(image_name)
    img_date_time = dt.datetime.strptime(image_name[0:image_name.find('(') - 3], '%Y-%m-%d-%H-%M-%S')  # drop ms
    image_dict.append({'Number': ii, 'Name': common_name, 'Year': img_date_time.year, 'Month': img_date_time.month,
                      'Day': img_date_time.day, 'Hour': img_date_time.hour, 'Seconds': img_date_time.second})
    # if ii > 10:
    #     break
df = pd.DataFrame(image_dict)
print(df.to_string())
name_counts = df['Name'].value_counts()
print(f'Name Counts: {name_counts}')
