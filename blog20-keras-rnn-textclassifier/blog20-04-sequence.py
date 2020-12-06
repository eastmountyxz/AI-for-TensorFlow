# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:01:09 2020

@author: xiuzhang
"""

from keras.preprocessing.sequence import pad_sequences

print(pad_sequences([[1, 2, 3], [1]], maxlen=2))
"""[[2 3] [0 1]]"""
print(pad_sequences([[1, 2, 3], [1]], maxlen=3, value=9))
"""[[1 2 3] [9 9 1]]"""

print(pad_sequences([[2,3,4]], maxlen=10))
"""[[0 0 0 0 0 0 0 2 3 4]]"""
print(pad_sequences([[1,2,3,4,5],[6,7]], maxlen=10))
"""[[0 0 0 0 0 1 2 3 4 5] [0 0 0 0 0 0 0 0 6 7]]"""

print(pad_sequences([[1, 2, 3], [1]], maxlen=2, padding='post'))
"""结束位置补: [[2 3] [1 0]]"""
print(pad_sequences([[1, 2, 3], [1]], maxlen=4, truncating='post'))
"""起始位置补: [[0 1 2 3] [0 0 0 1]]""" 

