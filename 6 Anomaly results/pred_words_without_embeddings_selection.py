# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 11:20:13 2022

@author: Amirhossein
"""

from pred_words_functions import find_nearest_words, load_glove_data, load_embeddings_data
import os
import json



knn = 10
measure_type = 'euclidean'

glove_vectors, glove_itos, glove_stoi = load_glove_data()
glove_data = (glove_vectors, glove_itos, glove_stoi)

y_pred, y_true, actual_names = load_embeddings_data()
predictions_data = y_pred, actual_names



final_result = find_nearest_words(glove_data,
                                  predictions_data,
                                  knn,
                                  measure_type)



save_folder = "results/word predictions without selection"
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

final_result_json = json.dumps(final_result)
save_path = os.path.join(save_folder, "final_result_{}_{}.json".format(measure_type, knn))
with open(save_path, "w") as f:
    f.write(final_result_json)






