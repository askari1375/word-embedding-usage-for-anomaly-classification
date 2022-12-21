# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:55:09 2022

@author: Amirhossein
"""

from annoy import AnnoyIndex
from tqdm import tqdm
import numpy as np
import json
import time


def load_glove_data():
    
    
    with open("GloVe Data/glove_stoi.json", "r") as f:
        glove_stoi = json.load(f)
    
    with open("GloVe Data/glove_itos.csv", 'r') as f:
        data = f.readlines()
    
    glove_itos = []
    for item in data:
        glove_itos.append(item[:-1])
    
        
    glove_vectors = np.load("GloVe Data/glove_vectors.npy")
    
    return glove_vectors, glove_itos, glove_stoi

def load_embeddings_data():
    
    y_pred = np.load("results/embeddings/anomaly predictions.npy")
    y_true = np.load("results/embeddings/anomaly true values.npy")
    
    actual_names = []
    with open("results/embeddings/names.txt", "r") as f:
        actual_names = f.readlines()
    
    return y_pred, y_true, actual_names



def find_nearest_words_for_one_vector(vector, k, ann_index, glove_itos):
    closest_item_indices = ann_index.get_nns_by_vector(vector, k)
    result = []
    for idx in closest_item_indices:
        word = glove_itos[idx]
        result.append(word)
        
    return result  



def find_nearest_words(glove_data,
                       predictions_data,
                       knn,
                       measure_type = 'euclidean'):

    num_trees = 50
    
    
    glove_vectors, glove_itos, glove_stoi = glove_data
    
    
    y_pred, actual_names = predictions_data
    
    embedding_size = glove_vectors.shape[1]
    ann_index = AnnoyIndex(embedding_size, measure_type)
    
    for idx in tqdm(range(len(glove_itos))):
        ann_index.add_item(idx, glove_vectors[idx, :])
    
    
    print("\nBuilding Annoy ...")
    start_time = time.time()
    ann_index.build(num_trees)
    duration = time.time() - start_time
    print("\nBuilding completed\t-\tDuration\t:\t{:.2f}".format(duration))
    
    
    # sample_vec = glove_vectors[glove_stoi["espresso"], :]
    # sample_result = find_nearest_words_for_one_vector(sample_vec, 10, ann_index, glove_itos)
    # print("Sample result:")
    # print(sample_result)
    
    
    
    final_result = {}
    
    for idx in tqdm(range(len(actual_names))):
        true_name = actual_names[idx]
        if true_name not in final_result.keys():
            final_result[true_name] = {}
      
        # fix me
        pred_emb = y_pred[idx, :]
        pred_words = find_nearest_words_for_one_vector(pred_emb, knn, ann_index, glove_itos)
    
        for pred_name in pred_words:
            if pred_name not in final_result[true_name].keys():
                final_result[true_name][pred_name] = 1
            else:
                final_result[true_name][pred_name] += 1
            
    
    return final_result




    




