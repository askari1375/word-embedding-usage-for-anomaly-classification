# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:33:30 2022

@author: Amirhossein
"""

import numpy as np
import matplotlib.pyplot as plt
from examin_loss_logs import extract_data
import os


def plot_histogram(data, total_rsme, plot_title):
    
    fig, ax = plt.subplots(figsize = (10, 8))
    
    ax.set_title(plot_title + " - RSME : {:.3f}".format(total_rsme))
    ax.hist(data, bins = 20)
    
    plt.tight_layout()
    
    save_folder = "results/plots"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    
    path = os.path.join(save_folder, "{}.png".format(plot_title))
    fig.savefig(path, dpi = 300)


def find_RSMEs(y_pred, y_true):
    
    d2 = np.square(y_pred - y_true)    
    total_rsme = np.sqrt(np.mean(d2))    
    axial_rsme = np.sqrt(np.mean(d2, axis = 0))
    
    return axial_rsme, total_rsme


def main():
    
    median_expantion_ratio = 0.8

    y_pred = np.load("results/embeddings/anomaly predictions.npy")
    y_true = np.load("results/embeddings/anomaly true values.npy")
    
    
    
    raw_axial_rsme, raw_total_rsme = find_RSMEs(y_pred, y_true)
        
    plot_histogram(raw_axial_rsme, raw_total_rsme, "Histogram without selection")
    
    print("Raw RSME\t:\t{}".format(raw_total_rsme))
    
    
    best_results, _ = extract_data()
    
    criterias = []
    for _ , c in best_results.values():
        criterias.append(c)
    
    criterias = np.array(criterias)
    
    median = np.median(criterias)
    indices = criterias <= median * median_expantion_ratio
    
    new_y_pred = y_pred[:, indices]
    new_y_true = y_true[:, indices]
    
    modified_axial_rsme, modified_total_rsme = find_RSMEs(new_y_pred, new_y_true)
        
    plot_histogram(modified_axial_rsme, modified_total_rsme, "Histogram with selection ({})".format(str(median_expantion_ratio)))
    
    print("Modified RSME\t:\t{}".format(modified_total_rsme))
    
if __name__ == "__main__":
    main()





