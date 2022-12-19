# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 14:33:24 2022

@author: Amirhossein
"""

from torch.utils.data import Dataset
import numpy as np
import os
import torch


class FeatureToEmbeddingDataset(Dataset):
    
    def __init__(self, embedding_idx, features_path, embeddings_path):
        
        self.embedding_idx = embedding_idx
        self.features_path = features_path
        self.embeddings_path = embeddings_path
        
        self.names_to_emb_dict = self.load_names_to_emb_dict()
        
        self.features_with_names = self.load_features()
        self.idx_map = self.generate_idx_map()
        
    
    
    def __getitem__(self, idx):
        
        features_with_names_idx, inter_class_idx = self.idx_map[idx]
        
        class_name = self.features_with_names[features_with_names_idx][0]
        feature = self.features_with_names[features_with_names_idx][1][inter_class_idx]
        
        emb = self.names_to_emb_dict[class_name]
        
        x = torch.tensor(feature, dtype = torch.float32)
        y = torch.tensor(emb[self.embedding_idx], dtype = torch.float32)
        
        return x, y
        
        
    
    def __len__(self):
        return len(self.idx_map)
    
    
    
    def generate_idx_map(self):
        idx_map = []
        
        for idx, (_, features) in enumerate(self.features_with_names):
            for k in range(features.shape[0]):
                idx_map.append((idx, k))
        
        return idx_map
    
    def load_features(self):
        
        result = []
        
        folder_names = os.listdir(self.features_path)
        
        for folder_name in folder_names:
            path = os.path.join(self.features_path,
                                folder_name,
                                "{}.npy".format(folder_name))
            features = np.load(path)
            result.append((folder_name, features))
            
        return result
    
    
    def load_names_to_emb_dict(self):
        
        
        
        final_names_path = os.path.join(self.embeddings_path, "final_names.txt")
        embs_path = os.path.join(self.embeddings_path, "final_embs.npy")
        
        with open(final_names_path, "r") as f:
            data = f.readlines()
        
        names = []
        for name in data:
            names.append(name[:-1])
        
        embs = np.load(embs_path)
                
        names_to_emb_dict = {}
        for idx in range(len(names)):
            names_to_emb_dict[names[idx]] = embs[idx, :]
        
        
        return names_to_emb_dict
        
        


def main():
    embedding_idx = 299
    
    embeddings_path = "D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Codes/3 feature to embedding dataset/results"
            
    features_parent_folder = "D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Codes/2 Image feature extraction - Tiny image net/extracted features/vit_base_patch16_224"
    features_path = os.path.join(features_parent_folder, "Normal/", "train/")
    
    dataset = FeatureToEmbeddingDataset(embedding_idx, features_path, embeddings_path)
    
    
    
    x, y = dataset[503]
    
    print("len dataset\t:\t{}".format(len(dataset)))
    
    print(x.shape)
    print(y)
    print(x)

if __name__ == "__main__":
    main()




