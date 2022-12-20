# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 21:03:46 2022

@author: Amirhossein
"""

import numpy as np
import matplotlib.pyplot as plt
import os



def get_log_loss_criteria(log_loss):
    val_loss = log_loss[1, :]
    best_val_loss = np.min(val_loss)
    return best_val_loss

def find_best_model_for_idx(path):
    
    model_names = os.listdir(path)
    
    best_criteria = np.inf
    best_model_name = None
    
    all_data = {}
    for model_name in model_names:
        model_folder = os.path.join(path, model_name)
        available_files = os.listdir(model_folder)
        log_loss = None        
        for file in available_files:
            if file[-3:] == "npy":
                log_loss = np.load(os.path.join(model_folder, file))
        if log_loss is not None:            
            criteria = get_log_loss_criteria(log_loss)
            if criteria < best_criteria:
                best_criteria = criteria
                best_model_name = model_name
        
        all_data[model_name] = criteria
    
    return best_model_name, best_criteria, all_data
                
            
        

def extract_data():


    emb_generators_folder = "D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Codes/4 Embedding generators/results"
    
    idx_file_names = os.listdir(emb_generators_folder)
    
    indices = []
    for name in idx_file_names:
        idx = int(name.split("_")[-1])
        indices.append(idx)
    
    idx_file_name_template = "_".join(name.split("_")[:-1])
    
    
    best_results_for_indices = {}
    
    models_results = {}
    
    for idx in indices:
        idx_file_name = idx_file_name_template + "_{}".format(idx)
        idx_file_name_path = os.path.join(emb_generators_folder, idx_file_name)
        best_model_for_idx, best_criteria_for_idx, all_data = find_best_model_for_idx(idx_file_name_path)
        
        new_model_name = "__".join(best_model_for_idx.split("__")[:-1])
        best_results_for_indices[idx] = (new_model_name, best_criteria_for_idx)
        
        for model_name in all_data.keys():
            model_name = "__".join(model_name.split("__")[:-1])
            if model_name not in models_results.keys():
                models_results[model_name] = {}
            models_results[model_name][idx] = all_data[model_name + "__{}".format(idx)]
            
    
    
    return best_results_for_indices, models_results


def plot_sorted_criteria(results):
    
    criterias = []
    for _ , c in results.values():
        criterias.append(c)
    
    criterias = np.array(criterias)
    criterias = np.sort(criterias)
    
    
    fig_criteri, ax_criteri = plt.subplots(figsize = (10, 8))
    
    
    line, = ax_criteri.plot(range(len(criterias)),
                        criterias,
                        color = 'black',
                        linestyle = "--",
                        )
    
    marker, = ax_criteri.plot(range(len(criterias)),
                        criterias,
                        markerfacecolor = 'tab:blue',
                        markeredgecolor = 'tab:red',
                        marker = 'o',
                        linestyle = 'None',
                        markersize = 15,
                        alpha = 0.5,
                        )
    
    ax_criteri.set_title("Best Validation Losses")
    ax_criteri.set_xlabel("Indices")
    ax_criteri.set_ylabel("Sorted Validation Loss")
    ax_criteri.set_xticks([])
    # ax_criteri.legend([line, marker], ['Best Validation Losses'])
    plt.grid('on')
    plt.tight_layout()
    fig_criteri.savefig('plots/Best Validation Losses.png', dpi = 300)
    

def main():

    best_results, models_results = extract_data()
    
    plot_sorted_criteria(best_results)


if __name__ == "__main__":
    main()