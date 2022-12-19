# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:47:40 2022

@author: Amirhossein
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


class ImageFeatureDataset(Dataset):
    
    def __init__(self, dataset_path):
                
        raw_data = pd.read_csv(dataset_path).to_numpy()
        
        features = raw_data[:, 0:-1]
        self.features = torch.tensor(features).float()
        labels = raw_data[:, -1]        
        self.labels = torch.tensor(labels).long()
        
        
        
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, index):
        
        x = self.features[index, :]
        y = self.labels[index]
        
        return x, y
    
    def get_data_shape(self):
        return self[0][0].shape

    def get_class_numbers(self):
        return np.unique(self.labels.numpy()).shape[0]


def main():
    
    dataset_folder_path = "D:/university/bioinformatics/Research/Word embedding usage for anomaly classification\Datasets/Tiny imagenet features"
    train_file_name = "train_tiny_features_ensemble_512_v5.csv"
    test_file_name = "test_tiny_features_ensemble_512_v5.csv"
    
    train_data_path = os.path.join(dataset_folder_path, train_file_name)
    test_data_path = os.path.join(dataset_folder_path, test_file_name)
    
    test_dataset = ImageFeatureDataset(test_data_path)
    print(len(test_dataset))
    
    x, y = test_dataset[9090]
    print(x.shape)
    print(y)
    
    print(test_dataset.get_data_shape())
    print(test_dataset.get_class_numbers())


if __name__ == "__main__":
    main()
    