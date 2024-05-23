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

# Data augmentation and normalization for training
# Just normalization for validation
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
    'train': transform,
    'val': transform
    
}

data_dir = '/nfs/uraskar/Data/high_res/behaviour_detection/batch_4/cow_id_settings/Experiment_5'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(dataset_sizes, class_names)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            batch_no = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if(batch_no % 100 == 0):
                    print('Batch no: ', batch_no)
                batch_no += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'id_behv5.pt')

        # print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights

    model.load_state_dict(best_model_wts)
    return model


model_ft = models.efficientnet_b0(pretrained=True)

# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

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
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Linear(num_ftrs, 16) # For Cow_id
model_ft.fc = nn.Linear(num_ftrs, 7) # For Behavior

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=8)
