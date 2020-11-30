# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 18:21:29 2019
@author: Eastmount CSDN YXZ
"""

import tensorflow as tf

# 传入值 给定type
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# 输出 乘法运算
output = tf.multiply(input1, input2)

# Session
with tf.Session() as sess:
    # placeholder需要传入值,在session.run时传入字典类型
    print(sess.run(output, feed_dict={input1:[7.], input2:[2.0]})) 
