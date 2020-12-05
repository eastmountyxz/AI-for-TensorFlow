# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 18:43:06 2020
@author: xiuzhang Eastmount CSDN
Wuhan fighting!
"""
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam

#------------------------------定义参数------------------------------
TIME_STEPS = 28     # 时间点数据 每次读取1行共28次 same as the height of the image 
INPUT_SIZE = 28     # 每行读取28个像素点 same as the width of the image
BATCH_SIZE = 50     # 每个批次训练50张图片
BATCH_INDEX = 0     
OUTPUT_SIZE = 10    # 每张图片输出分类矩阵
CELL_SIZE = 50      # RNN中隐藏单元
LR = 0.001          # 学习率

#---------------------------载入数据及预处理---------------------------
# 下载MNIST数据 
# training X shape (60000, 28x28), Y shape (60000, )
# test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 数据预处理
# 参数-1表示样例的个数 28*28表示像素长度和宽度
X_train = X_train.reshape(-1, 28, 28) / 255   # normalize
X_test = X_test.reshape(-1, 28, 28) / 255     # normalize

# 将类向量转化为类矩阵  数字 5 转换为 0 0 0 0 0 1 0 0 0 0 矩阵
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

#---------------------------创建RNN神经网络---------------------------
# 创建RNN模型
model = Sequential()

# RNN cell
model.add(SimpleRNN(
    # 设置输入batch形状 批次数量50 时间点28 每行读取像素28个
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape = (None, TIME_STEPS, INPUT_SIZE),
    # RNN输出给后一层的结果为50
    output_dim = CELL_SIZE,
    unroll=True,
))

# output layer
model.add(Dense(OUTPUT_SIZE))        # 全连接层 输出对应10分类
model.add(Activation('softmax'))     # 激励函数 tanh

#---------------------------神经网络优化器---------------------------
# optimizer
adam = Adam(LR)

# We add metrics to get more results you want to see
# 激活神经网络
model.compile(optimizer=adam,                      # 加速神经网络
              loss='categorical_crossentropy',     # 损失函数
              metrics=['accuracy'])                # 计算误差或准确率

#--------------------------------训练和预测------------------------------
cost_list = []
acc_list = []
step_list = []
for step in range(4001):
    # 分批截取数据 BATCH_INDEX初始值为0 BATCH_SIZE为50 取28个步长和28个INPUT_SIZE
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, : ]
    
    # 计算误差
    cost = model.train_on_batch(X_batch, Y_batch)
    
    # 累加参数 
    BATCH_INDEX += BATCH_SIZE
    # 如果BATCH_INDEX累加大于总体的个数 则重新赋值0开始分批计算
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
    
    # 每隔200步输出
    if step % 200 == 0:
        # 评价算法
        cost, accuracy = model.evaluate(
                X_test, y_test, 
                batch_size=y_test.shape[0], 
                verbose=False)
        # 写入列表
        cost_list.append(cost)
        acc_list.append(accuracy)
        step_list.append(step)
        print('test cost: ', cost, 'test accuracy: ', accuracy)
