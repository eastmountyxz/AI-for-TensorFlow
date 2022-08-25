# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:21:53 2021
@author: xiuzhang
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers import Convolution1D, MaxPool1D, Flatten, CuDNNLSTM
from keras.optimizers import RMSprop
from keras.layers import Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.models import Sequential
from keras.layers.merge import concatenate

import os
import tensorflow as tf
os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#指定了每个GPU进程中使用显存的上限,0.9表示可以使用GPU 90%的资源进行训练
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#---------------------------------------第一步 数据读取------------------------------------
#读取测数据集
train_df = pd.read_csv("weibo_3_fenci_train.csv")
val_df = pd.read_csv("weibo_3_fenci_val.csv")
test_df = pd.read_csv("weibo_3_fenci_test.csv")

#指定数据类型 否则AttributeError: 'float' object has no attribute 'lower' 存在文本为空的现象
print(train_df.head())

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体 SimHei黑体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'

#---------------------------------第二步 OneHotEncoder()编码---------------------------------
#对数据集的标签数据进行编码
train_y = train_df.label
print("Label:")
print(train_y[:10])
val_y = val_df.label
test_y = test_df.label
le = LabelEncoder()
train_y = le.fit_transform(train_y).reshape(-1,1)
print("LabelEncoder")
print(train_y[:10])    #99675
print(len(train_y))
val_y = le.transform(val_y).reshape(-1,1)
test_y = le.transform(test_y).reshape(-1,1)

#对数据集的标签数据进行one-hot编码
ohe = OneHotEncoder()
train_y = ohe.fit_transform(train_y).toarray()
val_y = ohe.transform(val_y).toarray()
test_y = ohe.transform(test_y).toarray()
print("OneHotEncoder:")
print(train_y[:10])

#-------------------------------第三步 使用Tokenizer对词组进行编码-------------------------------
#使用Tokenizer对词组进行编码
#当我们创建了一个Tokenizer对象后，使用该对象的fit_on_texts()函数，以空格去识别每个词,
#可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小。
max_words = 5000
max_len = 600
tok = Tokenizer(num_words=max_words)  ## 使用的最大词语数为5000
#print(train_df.cutword[:5])
tok.fit_on_texts(train_df.cutword)
print(tok)

#保存训练好的Tokenizer和导入
with open('tok.pickle', 'wb') as handle:
    pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('tok.pickle', 'rb') as handle:
    tok = pickle.load(handle)

#使用word_index属性可以看到每次词对应的编码
#使用word_counts属性可以看到每个词对应的频数
for ii,iterm in enumerate(tok.word_index.items()):
    if ii < 10:
        print(iterm)
    else:
        break
print("===================")  
for ii,iterm in enumerate(tok.word_counts.items()):
    if ii < 10:
        print(iterm)
    else:
        break
        
#使用tok.texts_to_sequences()将数据转化为序列
#使用sequence.pad_sequences()将每个序列调整为相同的长度
print(train_df.cutword)
data_train = train_df.cutword.astype(str)
data_val = val_df.cutword.astype(str)
data_test = test_df.cutword.astype(str)

train_seq = tok.texts_to_sequences(data_train)
val_seq = tok.texts_to_sequences(data_val)
test_seq = tok.texts_to_sequences(data_test)

#将每个序列调整为相同的长度
train_seq_mat = sequence.pad_sequences(train_seq,maxlen=max_len)
val_seq_mat = sequence.pad_sequences(val_seq,maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq,maxlen=max_len)
print(train_seq_mat.shape)  #(35000, 600)
print(val_seq_mat.shape)    #(5000, 600)
print(test_seq_mat.shape)   #(10000, 600)
print(train_seq_mat[:2])

#-------------------------------第四步 建立LSTM模型并训练-------------------------------
num_labels = 3
model = Sequential()
model.add(Embedding(max_words+1, 128, input_length=max_len))

#model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.1)))
model.add(Bidirectional(CuDNNLSTM(128)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_labels, activation='softmax'))
model.summary()
model.compile(loss="categorical_crossentropy",
              optimizer='adam',  # RMSprop()
              metrics=["accuracy"])
# 增加判断 防止再次训练
flag = "train"
if flag == "train":
    print("模型训练")
    ## 模型训练
    model_fit = model.fit(train_seq_mat, train_y, batch_size=128, epochs=10,
                          validation_data=(val_seq_mat,val_y),
                          callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)]  
                         )
    # 保存模型
    model.save('my_model.h5')  
    del model  # deletes the existing model
else:
    print("模型预测")
    # 导入已经训练好的模型
    model = load_model('my_model.h5')
    
    #--------------------------------------第五步 预测及评估--------------------------------
    #对测试集进行预测
    test_pre = model.predict(test_seq_mat)
    
    #评价预测效果，计算混淆矩阵
    confm = metrics.confusion_matrix(np.argmax(test_pre,axis=1),
                                     np.argmax(test_y,axis=1))
    print(confm)
    #混淆矩阵可视化
    Labname = ['喜悦','愤怒', '哀伤']
    print(metrics.classification_report(np.argmax(test_pre,axis=1),
                                        np.argmax(test_y,axis=1),
                                        digits=4))
    
    plt.figure(figsize=(8,8))
    sns.heatmap(confm.T, square=True, annot=True,
                fmt='d', cbar=False, linewidths=.6,
                cmap="YlGnBu")
    plt.xlabel('True label',size = 14)
    plt.ylabel('Predicted label', size = 14)
    plt.xticks(np.arange(3), Labname, size = 12)
    plt.yticks(np.arange(3), Labname, size = 12)
    plt.show()

    #--------------------------------------第六步 验证算法--------------------------------
    #使用tok对验证数据集重新预处理，并使用训练好的模型进行预测
    data_val = val_df.cutword.astype(str)
    val_seq = tok.texts_to_sequences(data_val)
    val_seq_mat = sequence.pad_sequences(val_seq,maxlen=max_len)
    val_pre = model.predict(val_seq_mat)
    print(metrics.classification_report(np.argmax(val_pre,axis=1),
                                        np.argmax(val_y,axis=1),
                                        digits=4))
