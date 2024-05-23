import os
import shutil
import math

## Code to create chunks and use differernt selections for different experiments
# Natural light chunks: 6 am to 6 pm: N1, N2, N3, N4, N5
# Artificial light chunks: 3 am to 6 am and 6 pm to 12 am: A1, A2, A3, A4, A5

# Part 1: Chunks are created - 5 Different selection arrangements called settings
# Settings are created on the available annotation files
# Input: Folder containing all annotations, from all cameras with names like cam1_filename.txt
# Naming this way allows us to use % len(listdir) for the splits, which is easier

def create_chunks(folder_path):
    # Get a sorted list of files
    files = sorted(os.listdir(folder_path))
    num_files = len(files)
    
    if num_files < 21:
        raise ValueError("The folder should contain at least 21 files to divide into chunks.")

    # Calculate number of files per part
    files_per_part = num_files / 21.0

    def get_indices(start, size):
        start = int(start)
        end = min(int(start + math.ceil(size)), num_files)
        return list(range(start, end))

    # Define chunk ranges (factor out files_per_part)
    chunk_ranges = {
        'N1': (3, 3 + 2.4),
        'N2': (5.4, 5.4 + 2.4),
        'N3': (7.8, 7.8 + 2.4),
        'N4': (10.2, 10.2 + 2.4),
        'N5': (12.6, 15),
        'A1': (15, 15 + 1.8),
        'A2': (16.8, 16.8 + 1.8),
        'A3': (18.6, 18.6 + 1.8),
        'A4': (20.4, 21),  # Wrap-around part 1
        'A4_part2': (0, 1.2),  # Wrap-around part 1 continued
        'A5': (1.2, 3)       # Wrap-around part 2
    }

    # Adjust ranges to indices
    chunks = {}
    for chunk, (start, end) in chunk_ranges.items():
        chunks[chunk] = get_indices(start * files_per_part, (end - start) * files_per_part)

    # Combine A4 parts
    chunks['A4'] = chunks.pop('A4') + chunks.pop('A4_part2')

    return chunks, files

def create_cross_validation_settings():
    train_val_test_order = [
        (['N1', 'N2', 'N3', 'A1', 'A2', 'A3'], ['N4', 'A4'], ['N5', 'A5']),
        (['N2', 'N3', 'N4', 'A2', 'A3', 'A4'], ['N5', 'A5'], ['N1', 'A1']),
        (['N3', 'N4', 'N5', 'A3', 'A4', 'A5'], ['N1', 'A1'], ['N2', 'A2']),
        (['N4', 'N5', 'N1', 'A4', 'A5', 'A1'], ['N2', 'A2'], ['N3', 'A3']),
        (['N5', 'N1', 'N2', 'A5', 'A1', 'A2'], ['N3', 'A3'], ['N4', 'A4']),
    ]
    return train_val_test_order

def copy_files_to_folders(settings, chunks, files, base_output_path, prefix):
    for i, (train_chunks, val_chunks, test_chunks) in enumerate(settings):
        setting_path = os.path.join(base_output_path, f"setting_{i + 1}")
        train_path = os.path.join(setting_path, "train", "labels")
        val_path = os.path.join(setting_path, "val", "labels")
        test_path = os.path.join(setting_path, "test", "labels")
        
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        
        def copy_files(chunk_names, dest_path):
            for chunk_name in chunk_names:
                for index in chunks[chunk_name]:
                    src_file_path = os.path.join(folder_path, files[index])
                    dest_file_path = os.path.join(dest_path, prefix + files[index])
                    shutil.copy2(src_file_path, dest_file_path)
        
        copy_files(train_chunks, train_path)
        copy_files(val_chunks, val_path)
        copy_files(test_chunks, test_path)

def print_settings(settings):
    for i, (train_chunks, val_chunks, test_chunks) in enumerate(settings):
        print(f"Setting {i + 1}:")
        print(f"  Train: {train_chunks}")
        print(f"  Val: {val_chunks}")
        print(f"  Test: {test_chunks}")
        print()

# Usage
for cam in ['cam_1', 'cam_2', 'cam_3', 'cam_4']:    
    folder_path = f"/nfs/hvvu2/CPS_Experimental_Data/Gopro_videos/labels/standing/0725/{cam}"
    base_output_path = "/nfs/uraskar/Data/high_res/new2_16class_data/new_settings"
    chunks, files = create_chunks(folder_path)
    settings = create_cross_validation_settings()
    print_settings(settings)
    copy_files_to_folders(settings, chunks, files, base_output_path, cam)


########

# Part 2: Once the annotations have been copied, copy images from all cameras into different folders


# Copy images
for setup in ['1','2','3','4','5']:
    for subfolder in ['train', 'val', 'test']:
        for cam in ['1','2','3','4']:
            labels_folder = f'/nfs/uraskar/Data/high_res/new2_16class_data/new_settings/setting_{setup}/{subfolder}/labels'
            images_folder = f'/nfs/hvvu2/CPS_Experimental_Data/Gopro_videos/images/15s_interval_images/images/0725/cam_{cam}'
            output_folder = f'/nfs/uraskar/Data/high_res/new2_16class_data/new_settings/setting_{setup}/{subfolder}/images'
            os.makedirs(output_folder, exist_ok=True)
            for file in os.listdir(labels_folder):
                if file.endswith('.txt'):
                    image_file = file.replace('.txt', '.jpg')
                    image_file = image_file.replace(f'cam_{cam}', '')
                    # print(image_file)
                    if os.path.isfile(os.path.join(images_folder, image_file)):
                        shutil.copy(os.path.join(images_folder, image_file), os.path.join(output_folder, f'cam_{cam}'+image_file))
                    else:
                        continue
        
        print(f'Done for setting_{setup} {subfolder}')
