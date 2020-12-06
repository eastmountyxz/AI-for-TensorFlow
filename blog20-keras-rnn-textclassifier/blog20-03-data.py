# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:31:03 2020
@author: xiuzhang
"""

from keras.datasets import imdb  

#获取数据
(x, y), _ = imdb.load_data(num_words=1)
print(x.shape, y.shape)

#特征词与ID映射
word2id = imdb.get_word_index()
id2word = {i: w for w, i in word2id.items()}
print(x[0])
print(' '.join([id2word[i] for i in x[0]]))
