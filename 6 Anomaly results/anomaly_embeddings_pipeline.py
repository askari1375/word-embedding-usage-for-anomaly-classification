# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 14:03:26 2022

@author: Amirhossein
"""


import torch
from torch.utils.data import DataLoader
from feature_to_embedding_dataset import FeatureToEmbeddingDataset
from neural_network import FullyConnectedNet
from examin_loss_logs import extract_data
from tqdm import tqdm
import numpy as np
import os




def get_prediction_for_data_loader(model, data_loader, device):
    
    true_result = []
    pred_result = []
    name_results = []
    
    with torch.no_grad():
        for x, y, names in tqdm(data_loader, disable = True):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            
            for element in torch.squeeze(y_pred):
                pred_result.append(element.to(torch.device("cpu")).numpy())
            for element in y:
                true_result.append(element.to(torch.device("cpu")).numpy())
            for name in names:
                name_results.append(name)
    
    
    true_result = np.array(true_result)
    pred_result = np.array(pred_result)
    
    return pred_result, true_result, name_results





def embedding_index_results(embedding_idx):


    """ --------------------------- Config --------------------------- """
    
    BATCH_SIZE = 64
    DATA_LOADER_KWARGS = {
        'batch_size' : BATCH_SIZE,
        'pin_memory' : True
        }
    
    
    weights_parent_folder = "D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Codes/4 Embedding generators/results"
    
    embeddings_path = "D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Codes/3 feature to embedding dataset/results"
                
    features_parent_folder = "D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Codes/2 Image feature extraction - Tiny image net/extracted features/vit_base_patch16_224/Anomaly"
    
    train_features_path = os.path.join(features_parent_folder, "train/")
    val_features_path = os.path.join(features_parent_folder, "val/")
    
    best_results, models_results = extract_data()
    
    
    """ --------------------------- Dataset --------------------------- """
    
    
    
    
    train_dataset = FeatureToEmbeddingDataset(embedding_idx,
                                              train_features_path,
                                              embeddings_path)
    
    val_dataset = FeatureToEmbeddingDataset(embedding_idx,
                                            val_features_path,
                                            embeddings_path)
    
    
    train_loader = DataLoader(train_dataset,
                              **DATA_LOADER_KWARGS)
    val_loader = DataLoader(val_dataset,
                            **DATA_LOADER_KWARGS)
    
    """ --------------------------- Load Model --------------------------- """
    
    best_model_name = best_results[embedding_idx][0]
    
    input_shape = int(best_model_name.split("__")[1])
    
    layer_str = best_model_name.split("__")[2]
    
    layer_nodes = [int(x) for x in layer_str.split("_")]
    
    network = FullyConnectedNet(input_shape, layer_nodes)
    
    weights_path = os.path.join(weights_parent_folder,
                                "emb_idx_{}".format(embedding_idx),
                                best_model_name + "__{}".format(embedding_idx),
                                best_model_name + "__{}.pt".format(embedding_idx))
    
    network.load_state_dict(torch.load(weights_path))
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    network.to(device)
    
    """ --------------------------- Prediction --------------------------- """
    
    final_results = {}
    
    final_results["train"] = get_prediction_for_data_loader(network, train_loader, device)
    final_results["val"] = get_prediction_for_data_loader(network, val_loader, device)
    
    
    
    return final_results


embedding_size = 300



sample_result = embedding_index_results(0)

n_train = len(sample_result["train"][2])
n_val = len(sample_result["val"][2])

train_pred = np.zeros((n_train, embedding_size))
train_true = np.zeros((n_train, embedding_size))
val_pred = np.zeros((n_val, embedding_size))
val_true = np.zeros((n_val, embedding_size))

train_names = sample_result["train"][2]
val_names = sample_result["val"][2]

for embedding_idx in tqdm(range(embedding_size)):
    results = embedding_index_results(embedding_idx)
    train_pred[:, embedding_idx] = results["train"][0]
    train_true[:, embedding_idx] = results["train"][1]
    val_pred[:, embedding_idx] = results["val"][0]
    val_true[:, embedding_idx] = results["val"][1]
    
    


pred = np.concatenate((train_pred, val_pred), axis = 0)
true_values = np.concatenate((train_true, val_true), axis = 0)
names = train_names + val_names

print(pred.shape)
print(true_values.shape)
print(len(names))

if not os.path.exists("results/embeddings/"):
    os.mkdir("results/embeddings/")

np.save("results/embeddings/anomaly predictions.npy", pred)
np.save("results/embeddings/anomaly true values.npy", true_values)

with open("results/embeddings/names.txt", "w") as f:
    f.write('\n'.join(names) + '\n')





