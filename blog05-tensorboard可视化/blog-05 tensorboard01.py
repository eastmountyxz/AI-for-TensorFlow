# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:51:40 2019
@author: xiuzhang CSDN Eastmount
"""
import tensorflow as tf

#---------------------------------定义神经层---------------------------------
# 函数：输入变量 输入大小 输出大小 激励函数默认None
def add_layer(inputs, in_size, out_size, activation_function=None):
    # 命名层
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            # 权重为随机变量矩阵
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')  #行*列
        with tf.name_scope('biases'):
            # 定义偏置 初始值增加0.1 每次训练中有变化
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')           #1行多列
        with tf.name_scope('Wx_plus_b'):
            # 定义计算矩阵乘法 预测值
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        # 激活操作
        if activation_function is None: 
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

#-----------------------------设置传入的值xs和ys-------------------------------
# 输入inputs包括x_input和y_input
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input') #x_data传入给xs
    ys = tf.placeholder(tf.float32,[None, 1], name='y_input')  #y_data传入给ys

#---------------------------------定义神经网络---------------------------------
# 隐藏层
L1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu) #激励函数
# 输出层
prediction = add_layer(L1, 10, 1, activation_function=None)

#------------------------------定义loss和train-------------------------------
with tf.name_scope('loss'):
    # 预测值与真实值误差 平均值->求和->平方(真实值-预测值)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 
                         reduction_indices=[1]))
with tf.name_scope('train'):
    # 训练学习 学习效率通常小于1 这里设置为0.1可以进行对比
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #减小误差

#------------------------------初始化和文件写操作-------------------------------
# 定义Session
sess = tf.Session()
# 整个框架加载到文件中,才能从文件中加载出来至浏览器中查看
writer = tf.summary.FileWriter('logs/', sess.graph)
# 初始化
init = tf.global_variables_initializer()
sess.run(init)
