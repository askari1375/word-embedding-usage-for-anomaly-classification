# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:58:52 2022

@author: Amirhossein
"""

import torch
from torch.utils.data import DataLoader
from fine_tuning_dataset import create_dataset
from train_functions import train
from my_network import MyNetwork
import os


""" --------------------------- Config --------------------------- """

num_epochs = 5

BATCH_SIZE = 16
learning_rate = 1e-3
DATA_LOADER_KWARGS = {
    'batch_size' : BATCH_SIZE,
    'shuffle' : True,
    'pin_memory' : True
    }

save_results_folder = "results/"
if not os.path.exists(save_results_folder):
    os.mkdir(save_results_folder)

""" --------------------------- Data Loader --------------------------- """

train_dataset, val_dataset = create_dataset()

train_loader = DataLoader(train_dataset,
                          **DATA_LOADER_KWARGS)
val_loader = DataLoader(val_dataset,
                        **DATA_LOADER_KWARGS)

input_shape = train_dataset[0][0].shape

""" --------------------------- Neural Network --------------------------- """

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = MyNetwork(len(train_dataset.classes))

model.get_summary(input_shape)

model_name = model.get_name()
save_folder = save_results_folder + model_name[:-3]
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)


""" ---------------------------- Train Process ---------------------------- """

model.to(device)

optimizer = torch.optim.Adam(model.get_parameters(), lr = learning_rate)

train(model,
      train_loader,
      val_loader,
      optimizer,
      num_epochs = num_epochs,
      device = device,
      save_folder = save_folder,
      model_name = model_name
      )







