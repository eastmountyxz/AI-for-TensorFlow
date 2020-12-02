# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:43:21 2020
@author: Eastmount
"""
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator

# 时间序列
y = np.array(range(5))
tg = TimeseriesGenerator(y, y, length=3, sampling_rate=1)
for i in zip(*tg[0]):
    print(*i)
