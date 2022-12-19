# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:37:45 2022

@author: Amirhossein
"""


from torchvision import transforms
from torchvision.datasets import ImageFolder

import os


def create_dataset():

    data_folder = 'D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Datasets/Tiny imagenet - processed dataset/Normal'
        
    
    train_dir = os.path.join(data_folder, 'train') 
    val_dir = os.path.join(data_folder, 'val')
    
    
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
                            ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
                            ])
    
    train_dataset = ImageFolder(train_dir,
                                transform = train_transforms)
    val_dataset = ImageFolder(val_dir,
                                transform = val_transforms)
    
    return train_dataset, val_dataset
    
def main():
    train_dataset, val_dataset = create_dataset()
    
    print(len(train_dataset))
    print(len(val_dataset))
    
    
if __name__ == "__main__":
    main()
