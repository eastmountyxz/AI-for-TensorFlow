# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:08:28 2020
@author: Eastmount CSDN
"""

from keras.datasets import imdb  #Movie Database
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.layers import LSTM

#-----------------------------------定义参数-----------------------------------
max_features = 20000       #按词频大小取样本前20000个词
input_dim = max_features   #词库大小 必须>=max_features
maxlen = 80                #句子最大长度
batch_size = 128           #batch数量
output_dim = 40            #词向量维度
epochs = 2                 #训练批次

#--------------------------------载入数据及预处理-------------------------------
#数据获取
(trainX, trainY), (testX, testY) = imdb.load_data(path="imdb.npz", num_words=max_features) 
print(trainX.shape, trainY.shape)  #(25000,) (25000,)
print(testX.shape, testY.shape)    #(25000,) (25000,)

#序列截断或补齐为等长
trainX = sequence.pad_sequences(trainX, maxlen=maxlen)
testX = sequence.pad_sequences(testX, maxlen=maxlen)
print('trainX shape:', trainX.shape)
print('testX shape:', testX.shape)

#------------------------------------创建模型------------------------------------
model = Sequential()

#词嵌入:词库大小、词向量维度、固定序列长度
model.add(Embedding(input_dim, output_dim, input_length=maxlen))

#平坦化: maxlen*output_dim
model.add(Flatten())

#输出层: 2分类
model.add(Dense(units=1, activation='sigmoid'))

#RMSprop优化器 二元交叉熵损失
model.compile('rmsprop', 'binary_crossentropy', ['acc'])

#训练
model.fit(trainX, trainY, batch_size, epochs)

#模型可视化
model.summary()


