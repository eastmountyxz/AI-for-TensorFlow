# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:38:31 2019
@author: Eastmount CSDN YXZ
"""

import tensorflow as tf

# 建立两个矩阵
matrix1 = tf.constant([[3,3]]) #常量 1行2列
matrix2 = tf.constant([[2],
                       [2]])  #常量 2行1列

# 矩阵乘法 matrix multiply 类似于numpy.dot()函数
product = tf.matmul(matrix1, matrix2)

# 两种利用Session会话控制的方法
# 方法一
sess = tf.Session()
output = sess.run(product) # 执行操作 每run一次TensorFlow才会执行操作
print(output)
sess.close() 

# 方法二
with tf.Session() as sess: # 打开Session并且赋值为sess 运行结束会自动close
    output = sess.run(product)
    print(output)
