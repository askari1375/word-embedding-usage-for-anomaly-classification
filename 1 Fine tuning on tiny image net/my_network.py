# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:14:37 2022

@author: Amirhossein
"""

import torch
import torch.nn as nn
import timm
from torchsummary import summary

class MyNetwork(nn.Module):
    
    def __init__(self, num_classes):
        
        super(MyNetwork, self).__init__()
        
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes = num_classes)
        model_parameters = list(self.model.parameters())
        for param in model_parameters[:-4]:
            param.requires_grad = False
            
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, x):        
        y = self.model(x)
        return y

    def get_loss(self, y_pred, y):
        loss = self.cross_entropy(y_pred, y)
        return loss
    
    def get_name(self):
        return 'vit_base_patch16_224.pt'
    
    def get_parameters(self):
        return self.parameters()
    
    def get_summary(self, input_size):
        summary(self, input_size, device = 'cpu')


def main():
    model = MyNetwork(5)
    model.get_summary((3, 224, 224))

if __name__ == "__main__":
    main()