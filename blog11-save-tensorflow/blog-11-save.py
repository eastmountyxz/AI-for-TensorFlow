# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 20:04:57 2020
@author: xiuzhang Eastmount CSDN
"""
import tensorflow as tf
import numpy as np

# 标记变量
train = False

#---------------------------------------保存文件---------------------------------------
# Save
if train==True:
    # 定义变量
    W = tf.Variable([[1,2,3], [3,4,5]], dtype=tf.float32, name='weights') #2行3列的数据
    b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

    # 初始化
    init = tf.global_variables_initializer()
    
    # 定义saver 存储各种变量
    saver = tf.train.Saver()
    
    # 使用Session运行初始化
    with tf.Session() as sess:
        sess.run(init)
        # 保存 官方保存格式为ckpt
        save_path = saver.save(sess, "my_net/save_net.ckpt")
        print("Save to path:", save_path)


#---------------------------------------Restore变量-------------------------------------
# Restore
if train==False:
    # 记住在Restore时定义相同的dtype和shape
    # redefine the same shape and same type for your variables
    W = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name='weights') #空变量
    b = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name='biases') #空变量
    
    # Restore不需要定义init
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 提取保存的变量
        saver.restore(sess, "my_net/save_net.ckpt")
        # 寻找相同名字和标识的变量并存储在W和b中
        print("weights", sess.run(W))
        print("biases", sess.run(b))

