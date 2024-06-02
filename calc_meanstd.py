import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from multiprocessing import Pool
import numpy as np
from PIL import Image
import time
import torch.utils.data as data_utils

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

training_transforms = transforms.Compose([
            #transforms.Resize((224, 224)),
            Custom_resize_transform(),
            transforms.ToTensor()
            ])

def get_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0
    #iteration = 0
    for images, labels in loader:
        #print(iteration)
        #iteration += 1
        #print(images.shape)
        images_count_in_batch = images.size(0)
        images = images.view(images_count_in_batch, images.size(1), -1)
        #print(images.shape)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += images_count_in_batch

    mean /= total_images_count
    std /= total_images_count

    return mean, std


# Calculate for all setting folders:
settings = ['1','2','3','4','5']

for setting in settings:
    # training_dataset_path = f'/nfs/uraskar/Data/high_res/behaviour_detection/batch_4/cropped_behavior/setting_{setting}/train'
    training_dataset_path = f'/nfs/uraskar/Data/high_res/behaviour_detection/omkar_copy/cow_id_folds4/lying/fold_{setting}/train'
    print(os.listdir(training_dataset_path))

    training_transforms = transforms.Compose([
                #transforms.Resize((224, 224)),
                Custom_resize_transform(),
                transforms.ToTensor()
                ])

    train_dataset = torchvision.datasets.ImageFolder(root=training_dataset_path, transform=training_transforms)
    # We will calculate it only on randomly selected 1k examples
    indices = torch.randperm(len(train_dataset))[:1000]
    train_dataset = data_utils.Subset(train_dataset, indices)
    # Comment above part later on
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)

    start_time = time.time()

    mean, std = get_mean_std(train_loader)
    print(f'Mean is {mean} | Std is {std} for setting {setting}')

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

