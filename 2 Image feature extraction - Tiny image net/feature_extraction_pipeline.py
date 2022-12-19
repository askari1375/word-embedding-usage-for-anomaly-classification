# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:35:11 2022

@author: Amirhossein
"""

import torch
from torch.utils.data import DataLoader
from feature_extractor import MyFeatureExtractor
from dataset import EachFolderDataset
import os
from tqdm import tqdm
import numpy as np
import csv



def extract_features_of_folder(feature_extractor, data_loader, device, features_shape):
    
    dataset_size = len(data_loader.dataset)
    results = np.zeros((dataset_size, features_shape))
    names = []
    
    c = 0
    
    with torch.no_grad():
        for x, y in tqdm(data_loader, disable=False):
            x = x.to(device)
            features = feature_extractor(x)
            features = features.to(torch.device("cpu")).numpy()
            results[c:c + x.shape[0], :] = features
            c += x.shape[0]
            for name in y:
                names.append([name])
    
    return results, names
    
    
        





""" --------------------------- Config --------------------------- """

batch_size = 16

features_shape = 768

save_results_folder = "extracted features/vit_base_patch16_224"
if not os.path.exists(save_results_folder):
    os.mkdir(save_results_folder)


""" --------------------------- Neural Network --------------------------- """

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
feature_extractor = MyFeatureExtractor()
feature_extractor.to(device)


""" --------------------------- Get Features --------------------------- """


data_folder = 'D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Datasets/Tiny imagenet - processed dataset/'


for p1 in ["Normal", "Anomaly"]:
    for p2 in ["train", "val"]:
        new_folder = os.path.join(save_results_folder, p1, p2)
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)

for p1 in ["Normal", "Anomaly"]:
    for p2 in ["train", "val"]:
        parent_folders = os.listdir(os.path.join(data_folder, p1, p2))
        for folder_name in parent_folders:
            new_folder = os.path.join(save_results_folder, p1, p2, folder_name)
            if not os.path.exists(new_folder):
                img_folder_path = os.path.join(data_folder, p1, p2, folder_name)
                dataset = EachFolderDataset(img_folder_path)
                data_loader = DataLoader(dataset, batch_size)

                features, names = extract_features_of_folder(feature_extractor, data_loader, device, features_shape)
                
                os.mkdir(new_folder)
                
                np.save(os.path.join(new_folder, folder_name + ".npy"), features)
                with open(os.path.join(new_folder, folder_name + "_image_names.csv"), 'w', newline='') as f:
                    write = csv.writer(f)
                    write.writerows(names)
    
          











                    
