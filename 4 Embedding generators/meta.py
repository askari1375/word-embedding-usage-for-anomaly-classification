# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:34:28 2022

@author: Amirhossein
"""

def create_model_name(input_shape,
                      layer_nodes,
                      emb_idx):
    
    
    model_name = "generator_"
    model_name += "_{}_".format(input_shape)
    for node in layer_nodes:
        model_name += "_{}".format(node)
    
    model_name += "__{}.pt".format(emb_idx)
    
    return model_name


def main():
    
    input_shape = 500
    layer_nodes = [256, 128, 64]
    emb_idx = 299
    
    model_name = create_model_name(input_shape, layer_nodes, emb_idx)
    
    print(model_name)
    
if __name__ == "__main__":
    main()