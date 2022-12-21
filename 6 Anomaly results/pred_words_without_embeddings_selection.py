# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 11:20:13 2022

@author: Amirhossein
"""

from annoy import AnnoyIndex
from tqdm import tqdm
import numpy as np
import json
import os

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


def find_nearest_words(vector, k, ann_index, glove_itos):
    closest_item_indices = ann_index.get_nns_by_vector(vector, k)
    result = []
    for idx in closest_item_indices:
        word = glove_itos[idx]
        result.append(word)
        
    return result    
    

glove_vectors, glove_itos, glove_stoi = load_glove_data()

y_pred, y_true, actual_names = load_embeddings_data()


num_trees = 50

measure_type = 'euclidean'

ann_index = AnnoyIndex(300, measure_type)

for idx in tqdm(range(len(glove_itos))):
    ann_index.add_item(idx, glove_vectors[idx, :])


print("\nBuilding annoy ...")
ann_index.build(num_trees)
print("Building completed")


sample_vec = glove_vectors[glove_stoi["espresso"], :]
sample_result = find_nearest_words(sample_vec, 10, ann_index, glove_itos)
print(sample_result)

knn = 10

final_result = {}

for idx in tqdm(range(len(actual_names))):
    true_name = actual_names[idx]
    if true_name not in final_result.keys():
        final_result[true_name] = {}
  
    # fix me
    pred_emb = y_pred[idx, :]
    pred_words = find_nearest_words(pred_emb, knn, ann_index, glove_itos)

    for pred_name in pred_words:
        if pred_name not in final_result[true_name].keys():
            final_result[true_name][pred_name] = 1
    else:
        final_result[true_name][pred_name] += 1


save_folder = "results/word predictions"
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

final_result_json = json.dumps(final_result)
save_path = os.path.join(save_folder, "final_result_{}_{}.json".format(measure_type, knn))
with open(save_path, "w") as f:
    f.write(final_result_json)






