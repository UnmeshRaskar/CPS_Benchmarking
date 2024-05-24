import os
import cv2
import time
from multiprocessing import Pool
from functools import partial

# This code will use GPU for faster cropping of bboxes from the video frames
# Scroll down to the end to make changes in just the directory paths

# Step 0) To crop bboxes from full frames using labels file
def process_image(image_file, image_folder, annotation_folder, output_folder):
    # Get the corresponding annotation file name
    annotation_file = os.path.join(annotation_folder, os.path.splitext(image_file)[0] + '.txt')
    if not os.path.exists(annotation_file):
        print(f"No corresponding annotation file found for image {image_file} and {annotation_file}")
        return

    # Read the image
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image '{image_path}'")
        return

    # Read the annotation file
    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_label = parts[0]
        # Convert string values to floats
        x_center, y_center, width, height = map(float, parts[1:])

        # There are 2 bbox formats, if this doesn't work, try the other one
        # Calculate the top-left and bottom-right corner coordinates of the bounding box
        x1 = int((x_center - width/2) * image.shape[1])
        y1 = int((y_center - height/2) * image.shape[0])
        x2 = int((x_center + width/2) * image.shape[1])
        y2 = int((y_center + height/2) * image.shape[0])

        # Ensure x1 and y1 are not negative # Some of the bounding boxes go outside the video frame image
        x1 = max(0, x1)
        y1 = max(0, y1)

        # Crop the region from the image
        cropped_region = image[y1:y2, x1:x2]

        # Create a subfolder for the class label if it doesn't exist
        class_folder = os.path.join(output_folder, class_label)
        os.makedirs(class_folder, exist_ok=True)

        # Save the cropped region to the class folder
        cropped_image_path = os.path.join(class_folder, f"{os.path.splitext(image_file)[0]}_{class_label}.jpg")
        cv2.imwrite(cropped_image_path, cropped_region)

def create_folders_and_crop(image_folder, annotation_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # List image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    print(f'Number of images: {len(image_files)}')

    # Use multiprocessing to process images in parallel
    with Pool() as pool:
        process_image_partial = partial(process_image, image_folder=image_folder, annotation_folder=annotation_folder, output_folder=output_folder)
        pool.map(process_image_partial, image_files)

###################

# We have created a directory structure like
'''
Setting_1 ----- Train --------- 1,2,3,...,16
                Val   --------- 1,2,3,...,16
                Test  --------- 1,2,3,...,16
Setting_2  ----- Train --------- 1,2,3,...,16
                Val   --------- 1,2,3,...,16
                Test  --------- 1,2,3,...,16
....
'''

subfolders = ['train', 'test', 'val']
settings = ['1','2','3','4', '5']

start_time = time.time()

# Change 1) Make changes to image_folder, annotation_folder, output_folder paths
for setting in settings:
    for subfolder in subfolders:
        image_folder = f"/nfs/uraskar/Data/high_res/new2_16class_data/new_settings/setting_{setting}/{subfolder}/images"
        annotation_folder = f"/nfs/uraskar/Data/high_res/new2_16class_data/new_settings/setting_{setting}/{subfolder}/labels"
        output_folder = f'/nfs/uraskar/Data/high_res/new2_16class_data/cropped_newsettings/setting_{setting}/{subfolder}'
        create_folders_and_crop(image_folder, annotation_folder, output_folder)
        print(f"Cropping completed for setting_{setting} {subfolder}")

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
