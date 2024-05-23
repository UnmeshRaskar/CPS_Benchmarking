import os

# Manually labeled classes are 1-16, yolo needs them to be 0-15
# Hence, we subtract one from each

# For training the Cow Detector, we are converting all labels to 0, ie. single class of Cow

for setup in ['1', '2','3','4','5']: #['2','3','4','5']
    for subfolder in ['train', 'val', 'test']:
        folder_path = f'/nfs/uraskar/Data/high_res/new_yolo_one_class/all_settings/setting_{setup}/{subfolder}/labels'

        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                with open(file_path, 'w') as file:
                    for line in lines:
                        parts = line.strip().split(' ')
                        if len(parts) > 0:
                            class_label = int(parts[0])
                            updated_label = 0 # class_label - 1
                            updated_line = str(updated_label) + ' ' + ' '.join(parts[1:]) + '\n'
                            file.write(updated_line)
        print('Done')

