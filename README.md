# CPS_Benchmarking
Benchmarking experiments for OpenCows2024

# Pipeline:
For the video frames available, we decided to use 
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
Clone the official YOLOv7 repository from here <br>
Images are settings/setting_1/images. Lables are in settings/setting_1/labels (labels adjusted to single_class instead of 16 class) <br>
Run adjust_labels.py to convert all annotation files to single class (class='0') <br>
Follow the instructions here for modifying the YOLOv7 repo for the finetuning experiments: article link to modify cfg, data and loss definitions for w6 model weight <br>
Training- run train_aux.py <br>
Inference- run test.py <br>

# 2.0) Cropping the bboxes
To train the classifier on cow identification, we crop the bboxes from the available annotated frames <br>
Images directory = <br>
Labels directory = <br>
Run faster_crop.py 

# 2) Cow Identification
We use EfficientNetv2_b0-S for finetuning it on our individual cows identification as a 16-class classifier <br>
Clone the official EfficientNet v2 implementation <br>
Modify the transforms, calculate the mean and std <br>
Use the calculated mean and std values to run train_effnet.py and then run test_effnet.py <br>


# 3) Cow Behavior Classification
We use EfficientNetv2_b0-S for finetuning it as a 7-class behavior classifier <br>
Clone the official EfficientNet v2 implementation <br>
Modify the transforms, calculate the mean and std <br>
Use the calculated mean and std values to run train_effnet.py and then run test_effnet.py <br>

We do two types of splitting for behavior classification:  <br>
Type 1 is the same as before (temporal), Type 2 is training on 12 cows, validating on 2 cows, testing on 2 other cows <br>
Example of Type 2 splitting. Setting 1- Train: Cow 1 to Cow 12, Val: Cow 13, Cow 14, Test: Cow 15, Cow 16 <br>
Setting 2- Train: Cow 3 to Cow 14, Val: Cow 15, Cow 16, Test: Cow 1, Cow 2 <br>
....

> Setting up the directories for Splitting Strategy_1 <br>
Individual cow behavior directory =                   <br>
Filtered behavior directory =                         <br>
Cropped bboxes directory =                            <br>
First, run preprocess_csv.py to filter the cow behavior csv files so that they are much smaller <br>
Run modified_crop_behavior.py <br>

> Setting up the directories for Splitting Strategy_2  <br>
Individual cow behavior directory =                    <br>
Filtered behavior directory =                          <br>
Cropped bboxes directory =                             <br> 
Run create_id_settings.py, then run cow_unit_split.py  <br>
