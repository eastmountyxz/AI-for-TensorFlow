# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:28:55 2019
@author: xiuzhang Eastmount CSDN
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 下载数据集 数字1到10
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#---------------------------------定义神经层---------------------------------
# 函数：输入变量 输入大小 输出大小 激励函数默认None
def add_layer(inputs, in_size, out_size, activation_function=None):
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
    return outputs

#-------------------------------定义计算准确度函数------------------------------
# 参数：预测xs和预测ys
def compute_accuracy(v_xs, v_ys):
    # 定义全局变量
    global prediction
    # v_xs数据填充到prediction变量中 生成预测值0到1之间的概率
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    # 比较预测最大值(y_pre)和真实最大值(v_ys)的差别 如果等于就是预测正确,否则错误
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    # 计算正确的数量
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 输出结果为百分比 百分比越高越准确
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result
    
#-------------------定义placeholder输入至神经网络---------------------------
# 设置传入的值xs和ys
xs = tf.placeholder(tf.float32, [None, 784])  #每张图片28*28=784个点
ys = tf.placeholder(tf.float32,[None, 10])    #每个样本有10个输出

#---------------------------------增加输出层---------------------------------
# 输入是xs 784个像素点 10个输出值 激励函数softmax常用于分类
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

#------------------------------定义loss和训练-------------------------------
# 预测值与真实值误差 平均值->求和->平方(真实值-预测值)
cross_entropyloss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), 
                     reduction_indices=[1]))  #loss
# 训练学习 学习效率通常小于1 这里设置为0.5可以进行对比
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropyloss) #减小误差

#-----------------------------------初始化-----------------------------------
# 定义Session
sess = tf.Session()
# 初始化
init = tf.initialize_all_variables()
sess.run(init)

#---------------------------------神经网络学习---------------------------------
for i in range(1000):
    # 提取一部分的xs和ys
    batch_xs, batch_ys = mnist.train.next_batch(100) #从下载好的数据集提取100个样本
    # 训练
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
    # 每隔50步输出一次结果
    if i % 50 == 0:
        # 计算准确度
        print(compute_accuracy(
                mnist.test.images, mnist.test.labels))
