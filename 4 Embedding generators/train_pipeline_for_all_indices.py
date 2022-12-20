# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:50:36 2022

@author: Amirhossein
"""

from train_pipeline import train_pipeline


 

num_epochs = 4

networks = [(256, 128), (512,), (256, 256, 256)]


for idx, layer_nodes in enumerate(networks):
    for embedding_idx in range(300):
        show_details = embedding_idx == 0
        
        print("\nNetwork Number\t:\t{}\tEmbedding idx\t:\t{}\n".format(idx + 1, embedding_idx))
        
        train_pipeline(embedding_idx, layer_nodes, num_epochs, show_details)


# =============================================================================
# idx = 2
# layer_nodes = networks[idx]
# 
# for embedding_idx in range(201, 300):
#     show_details = embedding_idx == 202
#     
#     print("\nNetwork Number\t:\t{}\tEmbedding idx\t:\t{}\n".format(idx + 1, embedding_idx))
#     
#     train_pipeline(embedding_idx, layer_nodes, num_epochs, show_details)
# =============================================================================
