# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:45:06 2022

@author: Amirhossein
"""

import torch
from torch.utils.data import DataLoader
from feature_to_embedding_dataset import FeatureToEmbeddingDataset
from neural_network import FullyConnectedNet
from train_functions import train
from meta import create_model_name
import numpy as np
import os


""" --------------------------- Config --------------------------- """


def train_pipeline(embedding_idx, layer_nodes, num_epochs, show_details = False):

    BATCH_SIZE = 64
    learning_rate = 1e-3
    DATA_LOADER_KWARGS = {
        'batch_size' : BATCH_SIZE,
        'shuffle' : True,
        'pin_memory' : True
        }
    
    
    
    np.random.seed(0)
    save_results_folder = "results/emb_idx_{}".format(embedding_idx)
    if not os.path.exists(save_results_folder):
        os.mkdir(save_results_folder)
    
    
    """ --------------------------- Data Loader --------------------------- """
    
    
    embeddings_path = "D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Codes/3 feature to embedding dataset/results"
            
    features_parent_folder = "D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Codes/2 Image feature extraction - Tiny image net/extracted features/vit_base_patch16_224/Normal"
    
    train_features_path = os.path.join(features_parent_folder, "train/")
    val_features_path = os.path.join(features_parent_folder, "val/")
    
    
    train_dataset = FeatureToEmbeddingDataset(embedding_idx,
                                              train_features_path,
                                              embeddings_path)
    
    val_dataset = FeatureToEmbeddingDataset(embedding_idx,
                                            val_features_path,
                                            embeddings_path)
    
    
    train_loader = DataLoader(train_dataset,
                              **DATA_LOADER_KWARGS)
    val_loader = DataLoader(train_dataset,
                            **DATA_LOADER_KWARGS)
    
    
    if show_details:
        x, y = next(iter(train_loader))
        
        print("Len train dataset\t:\t{}".format(len(train_dataset)))
        print("Len val dataset\t:\t{}".format(len(val_dataset)))
        
        print("Data Loader shapes:")
        print(x.shape)
        print(y.shape)
    
    
    """ --------------------------- Neural Network --------------------------- """
    
    sample_x, _ = train_dataset[0]
    input_shape = sample_x.shape[0]
    
    network = FullyConnectedNet(input_shape, layer_nodes)
    
    if show_details:
        network.get_summary((1, input_shape,))
    
    model_name = create_model_name(input_shape, layer_nodes, embedding_idx)
    save_folder = os.path.join(save_results_folder, model_name[:-3])
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    
    if show_details:
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




def main():

    embedding_idx = 0    
    layer_nodes = (512, 256)
    
    num_epochs = 2
    
    train_pipeline(embedding_idx, layer_nodes, num_epochs, show_details = True)


if __name__ == "__main__":
    main()








