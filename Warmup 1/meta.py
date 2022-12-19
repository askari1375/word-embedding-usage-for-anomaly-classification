# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:34:28 2022

@author: Amirhossein
"""

def create_model_name(input_shape,
                      layer_nodes,
                      num_classes):
    
    
    model_name = "classifier_"
    model_name += "_{}_".format(input_shape)
    for node in layer_nodes:
        model_name += "_{}".format(node)
    
    model_name += "__{}.pt".format(num_classes)
    
    return model_name


def main():
    
    input_shape = 500
    layer_nodes = [256, 128, 64]
    num_classes = 20
    
    model_name = create_model_name(input_shape, layer_nodes, num_classes)
    
    print(model_name)
    
if __name__ == "__main__":
    main()