'''
Split the data temporally chunkwise, train, val, --test--,
There is a buffer of 10% samples around test set to ensure no overlap/no adjacent examples between train and test

Input: Directory containing frames/labels/samples names chronologically
Output: A folder structure like: fold1 -> train, val, test, fold2 -> train, val, test, .....

Then directly train models on each fold, calculate the evaluation metrics and report the average score
'''

def split_data_temporal_kfold2(source_folder, base_output_folder, k=4, buffer_ratio=0.1):
    # List all files in the source folder and sort them by time
    files = os.listdir(source_folder)
    num_files = len(files)
    print(f'Number of files is {num_files}')

    buffer_size = int(num_files * buffer_ratio)
    fold_size = num_files // k

    # Iterate over k folds
    for fold in range(k):
        # Create fold-specific folders
        fold_folder = os.path.join(base_output_folder, f'fold_{fold+1}')
        train_folder = os.path.join(fold_folder, 'train')
        val_folder = os.path.join(fold_folder, 'val')
        test_folder = os.path.join(fold_folder, 'test')

        for folder in [train_folder, val_folder, test_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        start_idx = (fold * fold_size) % num_files
        end_idx = (start_idx + fold_size) % num_files

        # Determine indices for training set
        if start_idx < end_idx:
            train_files = files[start_idx:end_idx]
        else:
            train_files = files[start_idx:] + files[:end_idx]

        # Determine indices for validation set with buffer
        val_start = (end_idx + buffer_size) % num_files
        val_end = (val_start + fold_size) % num_files

        if val_start < val_end:
            val_files = files[val_start:val_end]
        else:
            val_files = files[val_start:] + files[:val_end]

        # Determine indices for test set with buffer
        test_start = (val_end + buffer_size) % num_files
        test_end = (test_start + fold_size) % num_files

        if test_start < test_end:
            test_files = files[test_start:test_end]
        else:
            test_files = files[test_start:] + files[:test_end]

        # Clear folders before copying files
        for folder in [train_folder, val_folder, test_folder]:
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))

        # Copy files to train, val, and test folders
        for file in train_files:
            shutil.copy(os.path.join(source_folder, file), os.path.join(train_folder, file))
        for file in val_files:
            shutil.copy(os.path.join(source_folder, file), os.path.join(val_folder, file))
        for file in test_files:
            shutil.copy(os.path.join(source_folder, file), os.path.join(test_folder, file))

        print(f'Temporal split of data completed for fold {fold+1}')

# Example usage
source_folder = '/nfs/uraskar/Data/high_res/new2_16class_data/labels'
base_output_folder = '/nfs/uraskar/Data/high_res/new2_16class_data/final_data'
split_data_temporal_kfold2(source_folder, base_output_folder)
