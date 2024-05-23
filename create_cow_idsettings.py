import os
import shutil
from sklearn.model_selection import KFold

def create_experiment_folders(base_path, output_path, experiment_number, train_cows, val_cows, test_cows):
    for phase, cows in zip(['train', 'val', 'test'], [train_cows, val_cows, test_cows]):
        for cow in cows:
            cow_path = os.path.join(output_path, f'Experiment_{experiment_number}', phase)
            behaviors = os.listdir(os.path.join(base_path, cow))
            for behavior in behaviors:
                src = os.path.join(base_path, cow, behavior)
                dst = os.path.join(cow_path, behavior)
                os.makedirs(dst, exist_ok=True)
                for item in os.listdir(src):
                    s = os.path.join(src, item)
                    d = os.path.join(dst, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)

def main():
    base_path = '/nfs/uraskar/Data/high_res/behaviour_detection/batch_4/cow_id_split'  # Change this to your actual data path
    output_path = '/nfs/uraskar/Data/high_res/behaviour_detection/batch_4/cow_id_settings'
    cows = ['1','2','3','4','5','6','7','8','9','10','11','14','15','16']#[f'{i+1}' for i in range(16)]
    kf = KFold(n_splits=5)

    for experiment_number, (train_indices, test_indices) in enumerate(kf.split(cows), 1):
        train_cows = [cows[i] for i in train_indices[:-2]]
        val_cows = [cows[i] for i in train_indices[-2:]]
        test_cows = [cows[i] for i in test_indices]
        
        print(f'Experiment {experiment_number}')
        print(f'Train: {train_cows}')
        print(f'Val: {val_cows}')
        print(f'Test: {test_cows}')
        
        create_experiment_folders(base_path, output_path, experiment_number, train_cows, val_cows, test_cows)

if __name__ == '__main__':
    main()