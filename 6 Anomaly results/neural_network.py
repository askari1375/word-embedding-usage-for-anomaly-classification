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
                 layer_nodes):
        
        super(FullyConnectedNet, self).__init__()
        
        layers = []
        
        last_layer_nodes = input_shape
        for node in layer_nodes:
            layers.append(nn.Linear(last_layer_nodes, node))
            last_layer_nodes = node
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(last_layer_nodes, 1))
        layers.append(nn.Tanh())
        
        self.layers = nn.Sequential(*layers)
        self.mse = nn.MSELoss()
    
    def forward(self, x):        
        y = self.layers(x)
        return y
    
    def get_summary(self, input_shape, device = "cpu"):
        summary(self, input_shape, device = device)


    def get_loss(self, y_pred, y):
        loss = self.mse(y_pred, y)
        return loss

def main():
    
    input_shape = 128
    layer_nodes = [64, 32]
    network = FullyConnectedNet(input_shape, layer_nodes)
    
    x = torch.randn(16, input_shape)
    y = network(x)
    
    print(x.shape)
    print(y.shape)
    
    
    
    network.get_summary((1, input_shape,))
    


if __name__ == "__main__":
    main()