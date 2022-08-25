# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 20:51:40 2021
@author: xiuzhang
"""
import numpy as np
from keras_bert import AdamWarmup, calc_train_steps

#生成随机数
train_x = np.random.standard_normal((1024, 100))
print(train_x)

#分批训练
total_steps, warmup_steps = calc_train_steps(
    num_example=train_x.shape[0],
    batch_size=32,
    epochs=10,
    warmup_proportion=0.1,
)

optimizer = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)
print(optimizer)