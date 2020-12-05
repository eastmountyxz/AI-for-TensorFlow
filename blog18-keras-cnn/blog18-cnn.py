# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:47:37 2020
@author: xiuzhang Eastmount CSDN
Wuhan Fighting!
"""
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

#---------------------------载入数据及预处理---------------------------
# 下载MNIST数据 
# training X shape (60000, 28x28), Y shape (60000, )
# test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 数据预处理
# 参数-1表示样例的个数 1表示灰度照片(3对应RGB彩色照片) 28*28表示像素长度和宽度
X_train = X_train.reshape(-1, 1, 28, 28) / 255   # normalize
X_test = X_test.reshape(-1, 1, 28, 28) / 255     # normalize

# 将类向量转化为类矩阵  数字 5 转换为 0 0 0 0 0 1 0 0 0 0 矩阵
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

#---------------------------创建第一层神经网络---------------------------
# 创建CNN
model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
# 第一层利用Convolution2D卷积 
model.add(Convolution2D(
    filters = 32,                   # 32个滤波器 
    nb_row = 5,                     # 宽度 
    nb_col = 5,                     # 高度
    border_mode = 'same',           # Padding method
    input_shape = (1, 28, 28),      # 输入形状 channels height width
))
# 增加神经网络激活函数
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
# 池化层利用MaxPooling2D
model.add(MaxPooling2D(
    pool_size = (2, 2),             # 向下取样
    strides = (2,2),                # 取样跳2个
    padding='same',                 # Padding method
))

#---------------------------创建第二层神经网络---------------------------
# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, 5, border_mode='same'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(pool_size = (2, 2), border_mode='same'))

#-----------------------------创建全连接层------------------------------
# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())                  # 将三维层拉直
model.add(Dense(1024))                # 全连接层
model.add(Activation('relu'))         # 激励函数

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))                  # 输出10个单位 
model.add(Activation('softmax'))      # 分类激励函数

#--------------------------------训练和预测------------------------------
# 优化器 optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
# 激活神经网络
model.compile(optimizer=adam,                      # 加速神经网络
              loss='categorical_crossentropy',     # 损失函数
              metrics=['accuracy'])                # 计算误差或准确率

print('Training')
model.fit(X_train, y_train, epochs=6, batch_size=64,)  # 训练次数及每批训练大小

print('Testing')
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

