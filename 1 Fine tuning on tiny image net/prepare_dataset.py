# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:56:22 2022

@author: Amirhossein
"""

import os
import shutil
from tqdm import tqdm
import numpy as np


def create_val_folder():

    DATA_DIR = 'D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Datasets/Tiny imagenet/tiny-imagenet-200' # Original images come in shapes of [3,64,64]
    
    # Define training and validation data paths
    VALID_DIR = os.path.join(DATA_DIR, 'val')
    
    new_val_folder = 'D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Datasets/Tiny imagenet - processed dataset/Normal/val'
    
    # Create separate validation subfolders for the validation images based on
    # their labels indicated in the val_annotations txt file
    val_img_dir = os.path.join(VALID_DIR, 'images')
    
    # Open and read val annotations text file
    fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    
    # Create dictionary to store img filename (word 0) and corresponding
    # label (word 1) for every line in the txt file (as key value pair)
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()
    
    # Display first 10 entries of resulting val_img_dict dictionary
    {k: val_img_dict[k] for k in list(val_img_dict)[:10]}
    
    
    # Create subfolders (if not present) for validation images based on label,
    # and move images into the respective folders
    for img, folder in tqdm(val_img_dict.items()):
        newpath = (os.path.join(new_val_folder, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        
        original = os.path.join(val_img_dir, img)
        target = os.path.join(newpath, img)
        if os.path.exists(original) and not os.path.exists(target):
            shutil.copyfile(original, target)


def create_train_folder():
    
    DATA_DIR = 'D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Datasets/Tiny imagenet/tiny-imagenet-200' # Original images come in shapes of [3,64,64]
    
    new_train_folder = 'D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Datasets/Tiny imagenet - processed dataset/Normal/train'
    
    
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    
    folders = os.listdir(TRAIN_DIR)
    
    for folder in tqdm(folders):
        image_names = os.listdir(os.path.join(TRAIN_DIR, folder, "images/"))
        new_folder_path = os.path.join(new_train_folder, folder)
        if not os.path.exists(new_folder_path):
            os.mkdir(new_folder_path)
        for img_name in image_names:
            origin = os.path.join(TRAIN_DIR, folder, "images/", img_name)
            target = os.path.join(new_folder_path, img_name)
            shutil.copyfile(origin, target)


def rename_folders():
    
    raw_folder_names_path = 'D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Datasets/Tiny imagenet/tiny-imagenet-200/words.txt'
    
    data_folder = 'D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Datasets/Tiny imagenet - processed dataset/Normal'
    
    
    
    with open(raw_folder_names_path, 'r') as f:
        data = f.readlines()
    
    names_dict = {}
    for line in data:
        words = line.split('\t')
        current_name = words[0]
        new_name_list = words[1].split(", ")
        new_name = ""
        for word in new_name_list:
            if word[-1] == "\n":
                new_name += word[:-1]
            else:
                new_name += word
            new_name += "_"
        new_name = new_name[:-1]
        
        names_dict[current_name] = new_name
    
    for folder_name in ["val/", "train/"]:
        current_folder = os.path.join(data_folder, folder_name)
        prev_folder_names = os.listdir(current_folder)
        for prev_folder_name in prev_folder_names:
            original = os.path.join(current_folder, prev_folder_name)
            target = os.path.join(current_folder, names_dict[prev_folder_name])
            os.rename(original, target)
    
def create_anomaly_folder():
    
    anomaly_ratio = 0.2
    data_folder = 'D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Datasets/Tiny imagenet - processed dataset'
    
    class_names = os.listdir(os.path.join(data_folder, "Normal", "val/"))
    
    indices = list(range(len(class_names)))
    np.random.shuffle(indices)
    split = int(len(indices) * anomaly_ratio)
    
    anomaly_indices = indices[:split]
    
    for anomaly_idx in tqdm(anomaly_indices):
        anomaly_folder_name = class_names[anomaly_idx]
        for folder_type in ["val", "train"]:
            origin = os.path.join(data_folder, "Normal/", folder_type, anomaly_folder_name)
            target = os.path.join(data_folder, "Anomaly/", folder_type, anomaly_folder_name)
            shutil.move(origin, target)
    
    
    
        

def main():
    pass
    

if __name__ == "__main__":
    main()

