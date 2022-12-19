# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:02:42 2022

@author: Amirhossein
"""

import torch
import torch.nn as nn
import classifier_net


class MyFeatureExtractor(nn.Module):
    
    def __init__(self):
        
        super(MyFeatureExtractor, self).__init__()
        
        weights_path = 'trained networks/vit_base_patch16_224/vit_base_patch16_224.pt'
        num_classes = 160
        
        
        self.model = classifier_net.MyNetwork(num_classes)
        self.model.load_state_dict(torch.load(weights_path))
        
        self.model.model.head = nn.Identity()
        
    
    
    def forward(self, x):
        return self.model(x)



def main():

    fe = MyFeatureExtractor()    
    x = torch.randn(2, 3, 224, 224)    
    y = fe(x)
    print(y.shape)


if __name__ == "__main__":
    main()

