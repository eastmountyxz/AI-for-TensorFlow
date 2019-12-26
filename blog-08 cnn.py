# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:27:01 2019
@author: xiuzhang Eastmount CSDN
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 下载数据集 数字1到10
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#-------------------------------定义计算准确度函数------------------------------
# 参数：预测xs和预测ys
def compute_accuracy(v_xs, v_ys):
    # 定义全局变量
    global prediction
    # v_xs数据填充到prediction变量中 生成预测值0到1之间的概率
    y_pre = sess.run(prediction, feed_dict={xs:v_xs,keep_prob: 1})
    # 比较预测最大值(y_pre)和真实最大值(v_ys)的差别 如果等于就是预测正确,否则错误
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    # 计算正确的数量
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 输出结果为百分比 百分比越高越准确
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
    return result


#---------------------------------定义权重和误差变量------------------------------
# 输入shape返回变量定义的参数
def weight_variable(shape):
    # 产生截断正态分布随机数
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    # 误差初始值定义为0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#---------------------------------定义卷积神经网络层------------------------------
# 定义二维CNN x表示输入值或图片的值 W表示权重
def conv2d(x, W):
    # 输入x表示整张图片的信息 权重W strides表示步长跨度 [1,x_movement,y_movement,1]
    # strides:一个长度为4的列表 第一个和最后一个元素为1 第二个为元素是水平x方向的跨度 第三个元素为垂直y方向跨度
    # padding包括两种形式 VALID和SAME
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

#------------------------------------定义POOLING---------------------------------
def max_pool_2x2(x):
    # Must have strides[0] = striders[3] = 1
    # x_movement和y_movement隔两个步长移动一次 从而压缩整幅图片的长和宽
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
#-----------------------------定义placeholder输入至神经网络-------------------------
# 设置传入的值xs和ys
xs = tf.placeholder(tf.float32, [None, 784])  #每张图片28*28=784个点
ys = tf.placeholder(tf.float32, [None, 10])   #每个样本有10个输出
# keeping probability
keep_prob = tf.placeholder(tf.float32)

# 形状修改
# xs包括了所有的图片样本 -1表示图片个数维度暂时不管(后续补充) 
# 28*28表示像素点 1表示信道(该案例图片黑白为1,彩色为3)
x_image = tf.reshape(xs, [-1,28,28,1])
print(x_image.shape) #[n_samples,28,28,1]
    
#-------------------------------增加神经层 conv1 layer------------------------------
# 定义权重
W_conv1 = weight_variable([5,5,1,32]) #patch 5*5, input size 1, output size 32
# 定义bias
b_conv1 = bias_variable([32]) #32个长度

# 搭建CNN的第一层
# 嵌套一个relu非线性化激励处理  计算 = x_image输入*权重 + 误差
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28*28*32
# POOLING处理
h_pool1 = max_pool_2x2(h_conv1)                          # output size 14*14*32


#-------------------------------增加神经层 conv2 layer------------------------------
# 定义权重
W_conv2 = weight_variable([5,5,32,64]) #patch 5*5, input size 32, output size 64
# 定义bias
b_conv2 = bias_variable([64]) #64个长度
# 搭建CNN的第二层
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14*14*64
# POOLING处理
h_pool2 = max_pool_2x2(h_conv2)                          # output size 7*7*64


#-------------------------------增加神经层 func1 layer------------------------------
# 定义权重 输入值为conv2 layer的输出值7*7*64 输出为1024
W_fc1 = weight_variable([7*7*64, 1024])
# 定义bias
b_fc1 = bias_variable([1024]) #1024个长度
# 将h_pool2输出值7*7*64转换为一维数据 [n_samples,7,7,64]->>[n_samples,7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) #-1表示样本数
# 乘法
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 解决过拟合
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#-------------------------------增加神经层 func2 layer------------------------------
# 定义权重 输入值为1024 输出为10对应10个数字
W_fc2 = weight_variable([1024, 10])
# 定义bias
b_fc2 = bias_variable([10])
# 预测 使用softmax计算概率
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#------------------------------定义loss和训练-------------------------------
# 预测值与真实值误差 平均值->求和->ys*log(prediction)
cross_entropyloss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), 
                     reduction_indices=[1]))  #loss
# 训练学习 学习效率设置为0.0001
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropyloss) #减小误差


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
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:0.5})
    # 每隔50步输出一次结果
    if i % 50 == 0:
        # 计算准确度
        print(compute_accuracy(
                mnist.test.images, mnist.test.labels))



