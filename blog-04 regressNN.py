# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 21:34:11 2019
@author: xiuzhang CSDN Eastmount
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

#---------------------------------构造数据---------------------------------
# 输入
x_data = np.linspace(-1, 1, 300)[:,np.newaxis]   #维度
# 噪声
noise = np.random.normal(0, 0.05, x_data.shape)  #平均值0 方差0.05
# 输出
y_data =np.square(x_data) -0.5 + noise

# 设置传入的值xs和ys
xs = tf.placeholder(tf.float32, [None, 1]) #x_data传入给xs
ys = tf.placeholder(tf.float32,[None, 1]) #y_data传入给ys

#---------------------------------定义神经网络---------------------------------
# 隐藏层
L1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu) #激励函数
# 输出层
prediction = add_layer(L1, 10, 1, activation_function=None)

#------------------------------定义loss和初始化-------------------------------
# 预测值与真实值误差 平均值->求和->平方(真实值-预测值)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 
                     reduction_indices=[1]))
# 训练学习 学习效率通常小于1 这里设置为0.1可以进行对比
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #减小误差

# 初始化
init = tf.initialize_all_variables()
# 运行
sess = tf.Session()
sess.run(init)

#---------------------------------可视化分析---------------------------------
# 定义图片框
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
# 散点图
ax.scatter(x_data, y_data)
# 连续显示
plt.ion()
plt.show()

#---------------------------------神经网络学习---------------------------------
# 学习1000次
n = 1
for i in range(1000):
    # 训练
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data}) #假设用全部数据x_data进行运算
    # 输出结果 只要通过place_holder运行就要传入参数
    if i % 50==0:
        #print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        try:
            # 忽略第一次错误 后续移除lines的第一个线段
            ax.lines.remove(lines[0])
        except Exception:
            pass
        # 预测
        prediction_value = sess.run(prediction, feed_dict={xs:x_data})
        # 设置线宽度为5 红色
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5) 
        # 暂停
        plt.pause(0.1)
        # 保存图片
        name = "test" + str(n) + ".png"
        plt.savefig(name)
        n =  n + 1
