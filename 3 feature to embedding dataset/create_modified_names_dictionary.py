# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 14:59:45 2022

@author: Amirhossein
"""



with open("results/names without embeddings.txt", "r") as f:
    data = f.readlines()


names = []
for name in data:
    names.append(name[:-1])

final_str = ""

for name in names:
    final_str += "{}\t=\t''\n".format(name)


with open("results/empty modified names.txt", "w") as f:
    f.write(final_str)

