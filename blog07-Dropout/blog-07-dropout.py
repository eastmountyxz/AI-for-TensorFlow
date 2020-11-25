# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:50:08 2019
@author: xiuzhang CSDN Eastmount
"""

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#---------------------------------载入数据---------------------------------
# 加载数据data和target
digits = load_digits()
X = digits.data
y = digits.target
# 转换y为Binarizer 如果y是数字1则第二个长度放上1
y = LabelBinarizer().fit_transform(y)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

#---------------------------------定义神经层---------------------------------
# 函数：输入变量 输入大小 输出大小 神经层名称 激励函数默认None
def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    # 权重为随机变量矩阵
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))    #行*列
    # 定义偏置 初始值增加0.1 每次训练中有变化
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)             #1行多列
    # 定义计算矩阵乘法 预测值
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # 激活操作
    if activation_function is None: 
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name+'/outputs', outputs)
    return outputs

#--------------------------------定义placeholder-------------------------------
# 设置传入的值xs和ys
xs = tf.placeholder(tf.float32, [None, 64])   #8*8=64个点
ys = tf.placeholder(tf.float32, [None, 10])   #每个样本有10个输出
# keeping probability
keep_prob = tf.placeholder(tf.float32)

#---------------------------------增加神经层---------------------------------
# 隐藏层 输入是8*8=64 输出是100 激励函数tanh
L1 = add_layer(xs, 64, 100, 'L1', activation_function=tf.nn.tanh)
# 输入是100 10个输出值 激励函数softmax常用于分类
prediction = add_layer(L1, 100, 10, 'L2', activation_function=tf.nn.softmax)

#------------------------------定义loss和训练-------------------------------
# 预测值与真实值误差
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), 
                     reduction_indices=[1]))  #loss
# 记录loss tensorboard显示变化曲线
tf.summary.scalar('loss', cross_entropy)
# 训练学习 学习效率通常小于1 这里设置为0.6可以进行对比
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy) #减小误差

#-----------------------------------初始化-----------------------------------
# 定义Session
sess = tf.Session()
# 合并所有summary
merged = tf.summary.merge_all()
# summary写入操作
train_writer = tf.summary.FileWriter('logs/train', sess.graph)
test_writer = tf.summary.FileWriter('logs/test', sess.graph)
# 初始化
init = tf.initialize_all_variables()
sess.run(init)

#---------------------------------神经网络学习---------------------------------
for i in range(1000):
    # 训练
    sess.run(train_step, feed_dict={xs:X_train, ys:y_train, keep_prob:0.5})
    # 每隔50步输出一次结果
    if i % 50 == 0:
        # 运行和赋值
        train_result = sess.run(merged,
                                feed_dict={xs:X_train, ys:y_train, keep_prob:1.0})
        test_result = sess.run(merged,
                               feed_dict={xs:X_test, ys:y_test, keep_prob:1.0})
        # 写入Tensorboard可视化
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)

