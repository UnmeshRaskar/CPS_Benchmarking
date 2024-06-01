# CPS_Benchmarking
Benchmarking experiments for MmCows

# 0.0) Creating all folds (temporal)
Main image dir = /nfs/hvvu2/ML/Data/visual_data/<br>
Cropped image dir = /nfs/hvvu2/ML/Data/visual_data/cropped_bboxes/<br>

Make comment/uncomment change as per the usecase on Line 40, no other change anywhere else <br>

0.0.1) Creating folds for detection | code is ready, but not uploaded here, image copying takes longer time <br>

0.0.2) Creating folds for behavior detection: run create_folds/behav_modified.py with this argument <br>
python behav_modified.py --data_splits_config_file /nfs/uraskar/Data/high_res/behaviour_detection/omkar_copy/config_s2.json --image_dir /nfs/hvvu2/ML/Data/visual_data/cropped_bboxes/behaviors --output_dir /nfs/uraskar/Data/high_res/behaviour_detection/batch_11 <br>

0.0.3) Creating folds for Standing cows identification : run create_folds/behav_modified.py with this argument<br>
python behav_modified.py --data_splits_config_file /nfs/uraskar/Data/high_res/behaviour_detection/omkar_copy/config_s2.json --image_dir /nfs/hvvu2/ML/Data/visual_data/cropped_bboxes/behaviors --output_dir /nfs/uraskar/Data/high_res/behaviour_detection/batch_11<br>

0.0.4) Creating folds for Lying cows identification : run create_folds/behav_modified.py with this argument<br>
python behav_modified.py --data_splits_config_file /nfs/uraskar/Data/high_res/behaviour_detection/omkar_copy/config_s2.json --image_dir /nfs/hvvu2/ML/Data/visual_data/cropped_bboxes/behaviors --output_dir /nfs/uraskar/Data/high_res/behaviour_detection/batch_11<br>

# Pipeline:

# 0) Temporal splitting of the data
We split the daily footage duration (3 AM to 12 AM = 21 hrs) into 10 chunks <br> 
5 of these chunks (N1, N2, .., N5) are for natural light. 5 other chunks (A1, A2, ..., A5) are of artificial light <br>
Natural light chunks are between 6 am to 6 pm and of size 2.4 hrs. Artificial light chunks are from (6 pm to 12 am) and (3 am to 6 am) and of size 1.8 hrs. <br>
We train on 3 natural + 3 artificial chunks, validate on 1 natural + 1 artificial chunks and test on 1 natural + 1 artificial chunks. <br>
Eg. Setting 1- Train: N1, N2, N3, A1, A2, A3, Val: N4, A4, Test: N5, A5 <br>
Setting 2- Train: N2, N3, N4, A2, A3, A4, Val: N5, A5, Test: N1, A1 <br>
....
We combine the evaluation metrics on all 5 settings and report the combined result <br>

labels can be found in labels_dir, images can be found in image_dir <br>
To create the 5 settings of experiments, run chunking.py <br>

# 1) Cow Detection
We use YOLOv7-w6 for finetuning it on our cows dataset for a single class (cow) object detection <br>
- Clone the official YOLOv7 repository from here <br>
  - Images are settings/setting_1/images  <br>
  - Lables are in settings/setting_1/labels (labels adjusted to single class instead of 16 classes) <br>
- Run adjust_labels.py to convert all annotation files to single class (class='0') <br>
- Follow the instructions here for modifying the YOLOv7 repo for the finetuning experiments: article link to modify cfg, data and loss definitions for w6 model weight <br>
- The above yolo code is at /nfs/uraskar/Data/high_res/new_yolo_one_class/yolov7/<br>
- Training- run train_aux.py <br>
  - Example usage with arguments: <br>
  python train_aux.py --workers 8 --device 0,1,2,3 --batch-size 16 --epochs 8 --img-size 1280 1280 --data data/coco_custom.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-w6_custom.yaml --name omkar_combined_1 --weights weights/yolov7-w6_training.pt
- Inference- run test.py <br>
  - Example usage with arguments: <br>
  python test.py --device 0 --batch-size 16 --conf 0.001 --iou 0.65 --data data/coco_custom.yaml --img 1280 --weights /nfs/uraskar/Data/high_res/new_yolo_one_class/yolov7/runs/train/omkar_combined_1/weights/best.pt

# 2.0) Cropping the bboxes
- To train the classifier on cow identification, we crop the bboxes from the available annotated frames <br>- 
- Images directory = /nfs/uraskar/Data/high_res/new2_16class_data/2_mins_exps/id_combined_settings/setting_{setting}/{subfolder}/images<br>
- Labels directory = /nfs/uraskar/Data/high_res/new2_16class_data/2_mins_exps/id_combined_settings/setting_{setting}/{subfolder}/labels<br>
- Run faster_crop.py (uses GPU-accelerated cropping operation over a large number of files)

# 2.1) Cow Identification
We use EfficientNetv2_b0-S for finetuning it on our individual cows identification as a 16-class classifier <br>
- Croped bboxes are at /nfs/uraskar/Data/high_res/new2_16class_data/2_mins_exps/combined_cropped
- Calculate the mean and std of each fold by running calc_meanstd.py <br>
- Use the calculated mean and std values to run train_effnet_b0.py and then run infer_effnet_b0.py <br>


# 3) Cow Behavior Classification

We do two types of splitting strategies for behavior classification:  <br>
- Type 1 is the same as before (temporal), Type 2 is training on 12 cows, validating on 2 cows, testing on 2 other cows <br>
- Example of Type 2 splitting
  - Setting 1- Train: Cow 1 to Cow 12, Val: Cow 13, Cow 14, Test: Cow 15, Cow 16 <br>
  - Setting 2- Train: Cow 3 to Cow 14, Val: Cow 15, Cow 16, Test: Cow 1, Cow 2 <br>
....

> Setting up the directories for Splitting Strategy_1 <br>
- Individual cow behavior directory = /nfs/uraskar/Data/high_res/behaviour_detection/batch_4/cow_behaviors                  <br>
- Filtered behavior directory = /nfs/uraskar/Data/high_res/behaviour_detection/batch_4/filtered_df                   <br>
- Cropped bboxes directory = /nfs/uraskar/Data/high_res/behaviour_detection/batch_4/cropped_behavior                    <br>
- First, run preprocess_csv.py to filter the cow behavior csv files so that they are much smaller <br>
- Run modified_crop_behavior.py to get the <br>

> Setting up the directories for Splitting Strategy_2  <br>
- Individual cow behavior directory = /nfs/uraskar/Data/high_res/behaviour_detection/batch_4/cow_behaviors                   <br>
- Filtered behavior directory = /nfs/uraskar/Data/high_res/behaviour_detection/batch_4/filtered_df                         <br>
- Cropped bboxes directory =                             <br> 
- Run cow_unit_split.py, then run create_cow_idsettings.py <br>

We use EfficientNetv2_b0-S for finetuning it as a 7-class behavior classifier. Note: make changes in the last Dense layer to make #neurons = #behaviors <br>
- Calculate the mean and std of each fold by running calc_meanstd.py <br>
- Use the calculated mean and std values to run train_effnet_b0.py and then run infer_effnet_b0.py <br>
