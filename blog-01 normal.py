# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

By：Eastmount CSDN YXZ 2019-11-28
"""

import tensorflow as tf 

#查询TensorFlow版本
print(tf.__version__)

#定义a和b为两个常量
a = tf.constant([1, 2], name="a")
b = tf.constant([2, 3], name="b")
print(a)
print(b)

#随机生成一个正态分布
output = tf.random.normal([5,3])
print(output)

#创建2个矩阵并进行相乘
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[1, 2],[3, 4]])
product = tf.matmul(matrix1,matrix2)
print(matrix1)
print(matrix2)
print(product)
print(product.numpy())
