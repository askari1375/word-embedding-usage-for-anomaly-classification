# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:51:41 2022

@author: Amirhossein
"""

from pred_words_functions import find_nearest_words, load_glove_data, load_embeddings_data
import numpy as np
from examin_loss_logs import extract_data
import matplotlib.pyplot as plt
import os
import json



median_expantion_ratio = 1

best_results, _ = extract_data()

criterias = []
for _ , c in best_results.values():
    criterias.append(c)

criterias = np.array(criterias)

median = np.median(criterias)
indices = criterias <= median * median_expantion_ratio

"""
    measure_type    :    "angular", "euclidean", "manhattan", "hamming", or "dot".

"""


knn = 5
measure_type = 'angular'

glove_vectors, glove_itos, glove_stoi = load_glove_data()
glove_vectors = glove_vectors[:, indices]
glove_data = (glove_vectors, glove_itos, glove_stoi)


y_pred, y_true, actual_names = load_embeddings_data()
y_pred = y_pred[:, indices]
y_true = y_true[:, indices]
predictions_data = y_pred, actual_names


final_result = find_nearest_words(glove_data,
                                  predictions_data,
                                  knn,
                                  measure_type)



save_folder = "results/word predictions with selection"
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

final_result_json = json.dumps(final_result)
median_expantion_ratio_str = "dot".join(str(median_expantion_ratio).split("."))
save_path = os.path.join(save_folder, "final_result_{}_{}__{}.json".format(measure_type,
                                                                           knn,
                                                                           median_expantion_ratio_str))
with open(save_path, "w") as f:
    f.write(final_result_json)




new_criterisas = criterias[indices]
new_criterisas = np.sort(new_criterisas)

print(new_criterisas.shape[0])
plt.plot(range(new_criterisas.shape[0]), new_criterisas, ".")