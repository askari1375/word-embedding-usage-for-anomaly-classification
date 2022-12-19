# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:37:45 2022

@author: Amirhossein
"""


from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch

import os



class EachFolderDataset(Dataset):
    
    def __init__(self, folder_path):
        
        self.folder_path = folder_path
        self.image_names = os.listdir(self.folder_path)
        
        
        
        
    
    def __len__(self):        
        return len(self.image_names)
    
    def __getitem__(self, idx):
        
        image_name = self.image_names[idx]
        image_path = os.path.join(self.folder_path, image_name)
        
        img = Image.open(image_path)
        
        tensor_img = transforms.Resize(256)(img)
        tensor_img = transforms.CenterCrop(224)(tensor_img)
        tensor_img = transforms.ToTensor()(tensor_img)        
        if tensor_img.shape[0] == 1:
            tensor_img = torch.cat((tensor_img, tensor_img, tensor_img), 0)
        
        tensor_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])(tensor_img)
        
        
        
        return tensor_img, image_name



def main():

    data_folder = 'D:/university/bioinformatics/Research/Word embedding usage for anomaly classification/Datasets/Tiny imagenet - processed dataset/Normal'
    
    train_dir = os.path.join(data_folder, 'train') 
    # val_dir = os.path.join(data_folder, 'val')
    
    train_folders = os.listdir(train_dir)
    
    folder_path = os.path.join(train_dir, train_folders[0])
    
    dataset = EachFolderDataset(folder_path)
    
    print(len(dataset))
    
    x, y = dataset[50]
    
    print(x.shape)
    print(y)
    
    
    

if __name__ == "__main__":
    main()
    
   
