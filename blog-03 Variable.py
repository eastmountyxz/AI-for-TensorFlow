# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 16:52:18 2019
@author: Eastmount CSDN YXZ
"""
import tensorflow as tf

# 定义变量 初始值为0 变量名字为counter(用于计数)
state = tf.Variable(0, name='counter')
print(state.name)
print(state)

# 定义常量
one = tf.constant(1)
print(one)

# 新变量
result = tf.add(state, one)

# 更新: result变量加载到state中 state当前变量即为result
update = tf.assign(state, result)

# Tensorflow中需要初始化所有变量才能激活
init = tf.global_variables_initializer() # must have if define variable

# Session
with tf.Session() as sess:
    sess.run(init)
    # 三次循环更新变量
    for _ in range(3):
        sess.run(update)
        print(sess.run(state)) #直接输出state没用 需要run
