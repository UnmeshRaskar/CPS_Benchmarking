from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image

import time
import os
import copy

cudnn.benchmark = True
import sklearn
from sklearn.metrics import classification_report


# Change 1)
# Everytime a new dataset is used for training, mean and std values need to be updated. These values are calculated on the TRAINING set with the help of \
# calc_mean_std.py
mean = [0.2419, 0.2216, 0.2201]
std = [0.2257, 0.2108, 0.2055]

class Custom_resize_transform(object):
    def __init__(self, output_size = (224, 224)):
        #assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
 
 
    def __call__(self, img):
 
        old_size = img.size # width, height
        ratio = float(self.output_size[0])/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        img = img.resize(new_size)
        # Paste into centre of black padded image
        new_img = Image.new("RGB", (self.output_size[0],self.output_size[1]))
        new_img.paste(img, ((self.output_size[0]-new_size[0])//2, (self.output_size[1]-new_size[1])//2))
        
        return new_img

transform = transforms.Compose([
            #transforms.Resize((224, 224)),
            Custom_resize_transform(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std=std),
        ])

data_transforms = {
    'test': transform
    
}

# Change 2) Change the data directory 
data_dir = '/nfs/uraskar/Data/high_res/behaviour_detection/batch_4/cow_id_settings/Experiment_5'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=8)
              for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
class_names = image_datasets['test'].classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(dataset_sizes, class_names)

def infer(model):
    #was_training = model.training
    model.eval()
    images_so_far = 0
    #fig = plt.figure()
    true_labels = []
    pred_labels= []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            #labels = labels.to(device)


            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            true_labels.extend(labels.tolist())
            pred_labels.extend(preds.tolist())

            if(i==0):
                print(true_labels, labels)
      
    return true_labels, pred_labels

model_ft = models.efficientnet_b0()



# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

# Recreate the classifier layer and seed it to the target device
model_ft.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=output_shape, # same number of output units as our number of classes
                    bias=True)).to(device)



#num_ftrs = model_ft.fc.in_features
num_ftrs = 1280

# Change 4) Make sure the final dense layer has #neurons = #classes
# model_ft.fc = nn.Linear(num_ftrs, 16) # For Cow_id classification
model_ft.fc = nn.Linear(num_ftrs, 7) # For Behavior Classification

model_ft = model_ft.to(device)
# multi-procecss code, this will use multi-gpus
if torch.cuda.is_available():
    model_ft = torch.nn.DataParallel(model_ft)
    model_ft.to(device)
else:
    model_ft.to(device)


model_ft.load_state_dict(torch.load('id_behv5.pt')) # Change 3) Change the saved model.pt name

true_labels, pred_labels = infer(model_ft)
# print(true_labels) # For debug
# print(pred_labels) # For debug

# funct_print(true_labels, pred_labels, class_names)
print(classification_report(y_true = true_labels, y_pred = pred_labels, target_names = class_names, digits=4))
