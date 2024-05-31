import pandas as pd
import os
# import cv2
import shutil
import time

def crop_behavior_cow(cow_id, cam_id, subfolder, setting):
    # cow_id = cow_id
    # cam = cam_id

    if cow_id in ['01','02','03','04','05','06','07','08','09']:
        cow_id = cow_id[-1]
        print(f'New Id is {cow_id}')

    image_folder = f"/nfs/oprabhune/MmCows/vision_data/images_15s_interval/cropped_bboxes/fold_{setting}/{subfolder}/{cow_id}"
    # annotations_folder = f'/nfs/uraskar/Data/high_res/new2_16class_data/new_settings/setting_1/{subfolder}/labels'
    output_folder = f'/nfs/oprabhune/MmCows/vision_data/images_15s_interval/behavior_exps/fold_{setting}/{subfolder}'

    # Process each row in the DataFrame
    for index, row in filtered_df2.iterrows():
        # Get the timestamp and datetime values from the row
        timestamp_value = row['timestamp']
        datetime_value = row['datetime']

        # Convert timestamp_value to integer
        timestamp_value_int = int(timestamp_value)

        # Convert datetime_value to string in the format HH-MM-SS
        datetime_value_str = datetime_value.strftime('%H-%M-%S')

        
        # If we have already cropped bboxes, we don't need to crop them again
        # We can just select the correpsonding bboxes from the right folders
        
        # Construct the file name
        file_name = f"cam_{cam_id}{timestamp_value_int}_{datetime_value_str}_{cow_id}.jpg"
        file_path = os.path.join(image_folder, file_name)
        # print(file_path)

        # print(f"File path for row {index}: {file_path}")
        if not os.path.exists(file_path):
            print(f"No corresponding cropped file found for {file_path}")
            continue

        behavior = row['behavior']
        # print(f'behavior is {behavior}')

        # For cow-wise splitting
        behavior_subfolder = os.path.join(output_folder, str(int(behavior)))
        os.makedirs(behavior_subfolder, exist_ok=True)
        # print(behavior_subfolder)

        image_name = file_name
        add = (f"{os.path.splitext(image_name)[0]}_b{int(behavior)}.jpg")
        # print(add)
        # Save the cropped region to the class folder
        copied_image_path = os.path.join(behavior_subfolder, add)
        # cv2.imwrite(cropped_image_path, cropped_region)
        shutil.copy(file_path, copied_image_path)
        # print(f'Cropped to {cropped_image_path}')

        print(f"Copied and saved region for '{behavior}' in image '{copied_image_path}'")

    return None


cows = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']
cams = [1, 2, 3, 4]
settings = ['1','2'] #['1','2','3','4','5']

start_time = time.time()

for setting in settings:
        
    for cow in cows:
        # Read the CSV file into a DataFrame
        filtered_df2 = pd.read_csv(f'/nfs/uraskar/Data/high_res/behaviour_detection/batch_5/filtered_df/filtered_df_C{cow}.csv')

        # Convert the 'datetime' column to datetime object
        filtered_df2['datetime'] = pd.to_datetime(filtered_df2['datetime'])

        # Get the timestamp and datetime values from a row in the filtered dataframe
        timestamp_value = filtered_df2.iloc[0]['timestamp']
        datetime_value = filtered_df2.iloc[0]['datetime']

        # Convert timestamp_value to integer
        timestamp_value_int = int(timestamp_value)

        # Convert datetime_value to string in the format HH-MM-SS
        datetime_value_str = datetime_value.strftime('%H-%M-%S')

        for subfolder in ['train','val','test']:
            for cam in cams:
                crop_behavior_cow(cow, cam, subfolder, setting)
                print(f'cam{cam} for cow{cow} {subfolder} setting {setting} done')
                # break


end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")