# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:45:06 2022

@author: Amirhossein
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import ImageFeatureDataset
from neural_network import FullyConnectedNet
from train_functions import train
from meta import create_model_name
import numpy as np
import os


""" --------------------------- Config --------------------------- """

num_epochs = 100

train_val_ratio = 0.8

BATCH_SIZE = 16
learning_rate = 1e-5
DATA_LOADER_KWARGS = {
    'batch_size' : BATCH_SIZE,
    'pin_memory' : True
    }




dataset_folder_path = "D:/university/bioinformatics/Research/Word embedding usage for anomaly classification\Datasets/Tiny imagenet features"
train_file_name = "train_tiny_features_ensemble_1536_v6.csv"

train_data_path = os.path.join(dataset_folder_path, train_file_name)



np.random.seed(0)
save_results_folder = "results/"
if not os.path.exists(save_results_folder):
    os.mkdir(save_results_folder)


""" --------------------------- Data Loader --------------------------- """

train_dataset = ImageFeatureDataset(train_data_path)

num_train = len(train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(num_train * train_val_ratio)
train_indices, val_indices = indices[:split], indices[split:]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(train_dataset,
                          sampler = train_sampler,
                          **DATA_LOADER_KWARGS)
val_loader = DataLoader(train_dataset,
                        sampler = val_sampler,
                        **DATA_LOADER_KWARGS)


x, y = next(iter(train_loader))


print("Data Loader shapes:")
print(x.shape)
print(y.shape)


""" --------------------------- Neural Network --------------------------- """

input_shape = train_dataset.get_data_shape()[0]
layer_nodes = []
num_classes = train_dataset.get_class_numbers()
network = FullyConnectedNet(input_shape, layer_nodes, num_classes)

network.get_summary((1, input_shape,))

model_name = create_model_name(input_shape, layer_nodes, num_classes)
save_folder = save_results_folder + model_name[:-3]
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)


print(model_name)

""" ---------------------------- Train Process ---------------------------- """


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

network.to(device)

optimizer = torch.optim.Adam(network.parameters(), lr = learning_rate)

train(network,
      train_loader,
      val_loader,
      optimizer,
      num_epochs = num_epochs,
      device = device,
      save_folder = save_folder,
      model_name = model_name
      )
















