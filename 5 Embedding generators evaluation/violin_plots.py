# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 12:10:42 2022

@author: Amirhossein
"""

import numpy as np
import matplotlib.pyplot as plt
from examin_loss_logs import extract_data


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value




best_results, models_results = extract_data()


data = {}


best_criterias = []
for _ , c in best_results.values():
    best_criterias.append(c)

data["Best"] = best_criterias

for model_name in models_results.keys():
    if model_name != "Best":
        new_model_name = model_name.split("__")[-1]
    else:
        new_model_name = "Best"
    
    if new_model_name not in data.keys():
        data[new_model_name] = []
    
    for c in models_results[model_name].values():
        data[new_model_name].append(c)

plot_data = []
for key in data.keys():
    plot_data.append(np.array(data[key]))

fig, ax =  plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
parts = ax.violinplot(plot_data, showmeans=False, showmedians=False, showextrema=False)


for idx, pc in enumerate(parts['bodies']):
    pc.set_facecolor(['red', 'blueviolet', 'aqua', 'lime'][idx])
    pc.set_edgecolor('black')
    pc.set_alpha(1)

n = len(data.keys())

quartile1 = np.zeros((n,))
medians = np.zeros((n,))
quartile3 = np.zeros((n,))

for k in range(n):
    quartile1[k], medians[k], quartile3[k] = np.percentile(plot_data[k], [25, 50, 75], axis=0)

whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(plot_data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)


ax.xaxis.set_tick_params(direction='out')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(np.arange(1, len(plot_data) + 1))
ax.set_xticklabels(data.keys())
ax.set_xlim(0.25, len(plot_data) + 0.75)
ax.set_xlabel('Model Name')
ax.set_ylabel('Validation Loss')
ax.set_title('Best Validation Loss Distribution')

plt.tight_layout()
plt.savefig("plots/violin plots.png", dpi = 400)

plt.show()

