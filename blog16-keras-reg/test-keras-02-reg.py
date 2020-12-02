# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:43:21 2020 
@author: Eastmount CSDN YXZ
O(∩_∩)O Wuhan Fighting!!!
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

#---------------------------创建散点数据---------------------------
# 输入
X = np.linspace(-1, 1, 200)
# 随机化数据
np.random.shuffle(X)
# 输出
y = 0.5*X + 2 + np.random.normal(0, 0.05, (200,)) #噪声平均值0 方差0.05
# 绘制散点图
# plt.scatter(X, y)
# plt.show()

# 数据集划分(训练集-测试集)
X_train, y_train = X[:160], y[:160]  # 前160个散点
X_test, y_test = X[160:], y[160:]    # 后40个散点

#----------------------------添加神经层------------------------------
# 创建模型
model = Sequential()

# 增加全连接层 输出个数和输入个数(均为1个)
model.add(Dense(output_dim=1, input_dim=1)) 

# 搭建模型 选择损失函数(loss function)和优化方法(optimizing method)
# mse表示二次方误差 sgd表示乱序梯度下降优化器
model.compile(loss='mse', optimizer='sgd')

#--------------------------------Traning----------------------------
print("训练")
# 学习300次
for step in range(301):
    # 分批训练数据 返回值为误差
    cost = model.train_on_batch(X_train, y_train)
    # 每隔100步输出误差
    if step % 100 == 0:
        print('train cost:', cost)

#--------------------------------Test-------------------------------
print("测试")
# 运行模型测试 一次传入40个测试散点
cost = model.evaluate(X_test, y_test, batch_size=40)
# 输出误差
print("test cost:", cost)
# 获取权重和误差 layers[0]表示第一个神经层(即Dense)
W, b = model.layers[0].get_weights()
# 输出权重和偏置
print("weights:", W)
print("biases:", b)

#------------------------------绘制预测图形-----------------------------
y_pred = model.predict(X_test)
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, "red")
plt.show()


