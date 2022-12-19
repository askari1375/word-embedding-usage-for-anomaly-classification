# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:28:43 2022

@author: Amirhossein
"""

import torch
import torch.nn as nn
from torchsummary import summary



class FullyConnectedNet(nn.Module):
    
    def __init__(self,
                 input_shape,
                 layer_nodes,
                 num_classes):
        
        super(FullyConnectedNet, self).__init__()
        
        layers = []
        
        last_layer_nodes = input_shape
        for node in layer_nodes:
            layers.append(nn.Linear(last_layer_nodes, node))
            last_layer_nodes = node
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(last_layer_nodes, num_classes))
        layers.append(nn.Softmax())
        
        self.layers = nn.Sequential(*layers)
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, x):        
        y = self.layers(x)
        return y
    
    def get_summary(self, input_shape, device = "cpu"):
        summary(self, input_shape, device = device)


    def get_loss(self, y_pred, y):
        loss = self.cross_entropy(y_pred, y)
        return loss

def main():
    
    input_shape = 128
    layer_nodes = [64, 32]
    num_classes = 10
    network = FullyConnectedNet(input_shape, layer_nodes, num_classes)
    
    x = torch.randn(16, input_shape)
    y = network(x)
    
    print(x.shape)
    print(y.shape)
    
    
    
    network.get_summary((1, input_shape,))
    


if __name__ == "__main__":
    main()