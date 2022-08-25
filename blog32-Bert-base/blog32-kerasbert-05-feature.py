# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 21:05:39 2021
@author: xiuzhang
"""
from keras_bert import extract_embeddings

model_path = 'xxx/yyy/uncased_L-12_H-768_A-12'
texts = ['all work and no play', 'makes jack a dull boy~']

embeddings = extract_embeddings(model_path, texts)
