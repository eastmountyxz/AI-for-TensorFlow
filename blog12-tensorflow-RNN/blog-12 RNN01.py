# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:50:33 2020
@author: xiuzhang Eastmount CSDN
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 下载手写数字图像数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 设置参数
learning_rate = 0.001     # 学习效率
train_iters = 100000      # 训练次数
batch_size = 128          # 自定义

n_inputs = 28             # MNIST 输入图像形状 28*28 黑白图片高度为1
n_steps = 28              # time steps 输入图像的28行数据
n_hidden_units = 128      # 神经网络隐藏层数量
n_classes = 10            # 分类结果 数字0-0


#-----------------------------定义placeholder输入-------------------------
# 设置传入的值xs和ys
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])  #每张图片28*28=784个点
y = tf.placeholder(tf.float32, [None, n_classes])          #每个样本有10个输出

# 定义权重 进入RNN前的隐藏层 输入&输出
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
}

# 定义偏置 进入RNN前的隐藏层 输入&输出
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ])),
}


#---------------------------------定义RNN-------------------------------
def RNN(X, weights, biases):
    # hidden layer for input to cell
    #######################################################
    # X (128 batch, 28 steps, 28 inputs) 28行*28列 
    # X ==> (128*28, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # 隐藏层 输入
    # X_in ==> (128batch*28steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # 二维数据转换成三维数据 
    # 注意：神经网络学习时要注意其形状如何变化
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units]) # 128个隐藏层
    
    # cell
    #######################################################
    # Cell结构 隐藏层数 forget初始偏置为1.0(初始时不希望forget)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # RNN会保留每一步计算的结果state
    # lstm cell is divided into two parts (c_state, m_state) 主线c_state 分线m_state
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    # RNN运算过程 每一步的输出都存储在outputs序列中
    # 常规RNN只有m_state LSTM包括c_state和m_state
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
    
    # hidden layer for output as final results
    #######################################################
    # 第三层加工最终的输出
    # 最终输出=Cell的输出*权重输出+偏置数据
    # states包含了主线剧情和分线剧情 states[1]表示分线剧情的结果 即为outputs[-1]最后一个输出结果
    results = tf.matmul(states[1], weights['out']) + biases['out']
    
    # 第二种方法
    # 解包 unpack to list [(batch, outputs)..] * steps
    #outputs = tf.unstack(tf.transpose(outputs, [1,0,2])) # states is the last outputs
    #results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


#---------------------------------定义误差和训练-------------------------------
pre = RNN(x, weights, biases)
# 预测值与真实值误差
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre, labels=y))
# 训练学习 学习效率设置为0.001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost) #梯度下降减小误差


# 预测正确个数
correct_pred = tf.equal(tf.argmax(pre, 1), tf.argmax(y, 1))
# 准确度
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



#---------------------------------初始化及训练-------------------------------
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    # 循环每次提取128个样本
    while step * batch_size < train_iters:
        # 从下载好的数据集提取128个样本
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 形状修改 [128, 28, 28]
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        # 训练
        sess.run([train_step], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        # 每隔20步输出结果
        if step % 20 == 0: # 20*128
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
            }))
        step += 1






