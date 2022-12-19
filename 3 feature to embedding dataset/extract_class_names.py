# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 19:07:38 2022

@author: Amirhossein
"""

import os

dataset_folder = "D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Datasets/Tiny imagenet - processed dataset"


all_names = []


for p1 in ["Normal", "Anomaly"]:
    for p2 in ["train", "val"]:
        folder_path = os.path.join(dataset_folder, p1, p2)
        names = os.listdir(folder_path)
        for name in names:
            if name not in all_names:
                all_names.append(name)
        
with open("results/all names.txt", "w") as f:
    for name in all_names:
        f.write("{}\n".format(name))
