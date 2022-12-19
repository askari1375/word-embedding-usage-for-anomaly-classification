# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 20:12:52 2021

@author: Amirhossein
"""

import torch
import numpy as np
from my_network import MyNetwork
from fine_tuning_dataset import create_dataset
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


class NetworkEvaluator():
    def __init__(self,
                 num_classes):
        
        save_results_folder = "results/"
        self.num_classes = num_classes
        
        self.model = MyNetwork(num_classes)
        
        model_name = self.model.get_name()
        self.final_folder = os.path.join(save_results_folder, model_name[:-3])
        model_path = os.path.join(self.final_folder, model_name)
        self.model.load_state_dict(torch.load(model_path))
        
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        
        

    
    
    def get_prediction(self, idx, dataset):
        x, y = dataset[idx]
        x = torch.unsqueeze(x, dim = 0).to(self.device)
        #y = torch.unsqueeze(y, dim = 0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x)
        outputs = outputs.to(torch.device("cpu")).numpy()
        y_pred = np.argmax(outputs, axis=-1)
        
        #y_true = y.to(torch.device("cpu")).numpy().reshape((-1))
        y_true = y
        
        return y_pred, y_true
    
    def find_performance(self, dataset):
        
        n = len(dataset)

        y_pred = np.zeros((n,))
        y_true = np.zeros((n,))
        
        for idx in tqdm(range(n)):
             y_pred[idx], y_true[idx] = self.get_prediction(idx, dataset)
        
        accuracy = np.sum(y_pred == y_true) / n
        cm = confusion_matrix(y_true, y_pred)
       
        
        return accuracy, cm
    

    



image_shape = (224, 224)

train_dataset, val_dataset = create_dataset()

num_classes = len(train_dataset.classes)

network_evaluator = NetworkEvaluator(num_classes)




# =============================================================================
# train_accuracy, train_matrix = network_evaluator.find_performance(train_dataset)
# validation_accuracy, validation_matrix = network_evaluator.find_performance(val_dataset)
# =============================================================================

validation_accuracy, validation_matrix = network_evaluator.find_performance(val_dataset)
train_accuracy, train_matrix = validation_accuracy, validation_matrix 


# =============================================================================
# 
# report = ""
# report += "train accuracy\t:\t{}\n".format(train_accuracy)
# report += "validation accuracy\t:\t{}\n".format(validation_accuracy)
# 
# with open(network_evaluator.final_folder + "/accuracy.txt", 'w') as f:
#     f.write(report)
# 
# print(report)
# =============================================================================
