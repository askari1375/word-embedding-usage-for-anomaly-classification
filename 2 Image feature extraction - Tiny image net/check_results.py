# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 13:20:30 2022

@author: Amirhossein
"""

import numpy as np
import os

folder = "extracted features/vit_base_patch16_224"

npy_paths = []
for (root,dirs,files) in os.walk(folder, topdown=True):    
    for file in files:
        if file[-3:] == "npy":
            npy_paths.append(os.path.join(root, file))


print(len(npy_paths))

idx = -1

data = np.load(npy_paths[idx])

print(data.shape)
print(data)