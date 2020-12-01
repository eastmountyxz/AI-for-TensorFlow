# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:35:47 2020
@author: xiuzhang Eastmount CSDN
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#-----------------------------------初始化设置---------------------------------------
# 基础参数设置
learning_rate = 0.001   #学习效率
training_epochs = 20    #20组训练
batch_size = 256        #batch大小
display_step = 1
examples_to_show = 10   #显示10个样本

# 神经网络输入设置
n_input = 784           #MNIST输入数据集(28*28)

# 输入变量(only pictures)
X = tf.placeholder("float", [None, n_input])

# 隐藏层设置
n_hidden_1 = 128        #第一层特征数量
n_hidden_2 = 64         #第二层特征数量
n_hidden_3 = 10         #第三层特征数量
n_hidden_4 = 2          #第四层特征数量

weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
        'decoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h4': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
        'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b4': tf.Variable(tf.random_normal([n_input])),
}

# 导入MNIST数据
mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

#---------------------------------压缩和解压函数定义---------------------------------------
# Building the encoder
def encoder(x):
    # 压缩隐藏层调用函数sigmoid(压缩值为0-1范围内)
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    # 输出范围为负无穷大到正无穷大 调用matmul函数
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                    biases['encoder_b4'])
    return layer_4

# Building the decoder
def decoder(x):
    # 解压隐藏层调用sigmoid激活函数(范围内为0-1区间)
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                biases['decoder_b4']))
    return layer_4

#-----------------------------------压缩和解压操作---------------------------------------
# Construct model
# 压缩：784 => 128
encoder_op = encoder(X)

# 解压：784 => 128
decoder_op = decoder(encoder_op)

#--------------------------------对比预测和真实结果---------------------------------------
# 预测
y_pred = decoder_op

# 输入数据的类标(Labels)
y_true = X

# 定义loss误差计算 最小化平方差
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#-------------------------------------训练及可视化-------------------------------------
# 初始化
init = tf.initialize_all_variables()

# 训练集可视化操作
with tf.Session() as sess:  
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    
    # 训练数据
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x)=1 min(x)=0
            # 运行初始化和误差计算操作
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # 每个epoch显示误差值
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")
    
    # 观察解压前的结果
    encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
    # 显示encoder压缩成2个元素的预测结果
    plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
    plt.colorbar()
    plt.show()
