"""
Created on 2021-03-19
@author: xiuzhang Eastmount CSDN
LSTM Model
"""
import os
import time
import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers import Convolution1D, MaxPool1D, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.models import Sequential

#GPU加速 CuDNNLSTM比LSTM快
from keras.layers import CuDNNLSTM, CuDNNGRU

## GPU处理 读者如果是CPU注释该部分代码即可
## 指定每个GPU进程中使用显存的上限 0.9表示可以使用GPU 90%的资源进行训练
os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

start = time.clock()

#----------------------------第一步 数据读取----------------------------
## 读取测数据集
train_df = pd.read_csv("news_dataset_train_fc.csv")
val_df = pd.read_csv("news_dataset_val_fc.csv")
test_df = pd.read_csv("news_dataset_test_fc.csv")
print(train_df.head())

## 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  #指定默认字体 SimHei黑体
plt.rcParams['axes.unicode_minus'] = False   #解决保存图像是负号'

#--------------------------第二步 OneHotEncoder()编码--------------------
## 对数据集的标签数据进行编码
train_y = train_df.label
val_y = val_df.label
test_y = test_df.label
print("Label:")
print(train_y[:10])

le = LabelEncoder()
train_y = le.fit_transform(train_y).reshape(-1,1)
val_y = le.transform(val_y).reshape(-1,1)
test_y = le.transform(test_y).reshape(-1,1)
print("LabelEncoder")
print(train_y[:10])
print(len(train_y))

## 对数据集的标签数据进行one-hot编码
ohe = OneHotEncoder()
train_y = ohe.fit_transform(train_y).toarray()
val_y = ohe.transform(val_y).toarray()
test_y = ohe.transform(test_y).toarray()
print("OneHotEncoder:")
print(train_y[:10])

#-----------------------第三步 使用Tokenizer对词组进行编码--------------------
max_words = 6000
max_len = 600
tok = Tokenizer(num_words=max_words)  #最大词语数为6000
print(train_df.cutword[:5])
print(type(train_df.cutword))

## 防止语料中存在数字str处理
train_content = [str(a) for a in train_df.cutword.tolist()]
val_content = [str(a) for a in val_df.cutword.tolist()]
test_content = [str(a) for a in test_df.cutword.tolist()]
tok.fit_on_texts(train_content)
print(tok)

## 保存训练好的Tokenizer和导入
with open('tok.pickle', 'wb') as handle: #saving
    pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('tok.pickle', 'rb') as handle: #loading
    tok = pickle.load(handle)

#---------------------------第四步 数据转化为序列-----------------------------
train_seq = tok.texts_to_sequences(train_content)
val_seq = tok.texts_to_sequences(val_content)
test_seq = tok.texts_to_sequences(test_content)

## 将每个序列调整为相同的长度
train_seq_mat = sequence.pad_sequences(train_seq,maxlen=max_len)
val_seq_mat = sequence.pad_sequences(val_seq,maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq,maxlen=max_len)
print("数据转换序列")
print(train_seq_mat.shape)
print(val_seq_mat.shape)
print(test_seq_mat.shape)
print(train_seq_mat[:2])

#-------------------------------第五步 建立LSTM模型--------------------------
## 定义LSTM模型
inputs = Input(name='inputs',shape=[max_len],dtype='float64')
## Embedding(词汇表大小,batch大小,每个新闻的词长)
layer = Embedding(max_words+1, 128, input_length=max_len)(inputs)
#layer = LSTM(128)(layer)
layer = CuDNNLSTM(128)(layer)

layer = Dense(128, activation="relu", name="FC1")(layer)
layer = Dropout(0.1)(layer)
layer = Dense(4, activation="softmax", name="FC2")(layer)
model = Model(inputs=inputs, outputs=layer)
model.summary()
model.compile(loss="categorical_crossentropy",
              optimizer='adam',  # RMSprop()
              metrics=["accuracy"])

#-------------------------------第六步 模型训练和预测--------------------------
## 先设置为train训练 再设置为test测试
flag = "test"
if flag == "train":
    print("模型训练")
    ## 模型训练 当val-loss不再提升时停止训练 0.0001
    model_fit = model.fit(train_seq_mat, train_y, batch_size=128, epochs=10,
                          validation_data=(val_seq_mat,val_y),
                          callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)]  
                         )
    model.save('my_model.h5')  
    del model
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)
    print(model_fit.history)
    
else:
    print("模型预测")
    ## 导入已经训练好的模型
    model = load_model('my_model.h5')
    ## 对测试集进行预测
    test_pre = model.predict(test_seq_mat)
    ## 评价预测效果，计算混淆矩阵
    confm = metrics.confusion_matrix(np.argmax(test_y,axis=1),np.argmax(test_pre,axis=1))
    print(confm)
    
    ## 混淆矩阵可视化
    Labname = ["体育", "文化", "财经", "游戏"]
    print(metrics.classification_report(np.argmax(test_y,axis=1),np.argmax(test_pre,axis=1)))
    plt.figure(figsize=(8,8))
    sns.heatmap(confm.T, square=True, annot=True,
                fmt='d', cbar=False, linewidths=.6,
                cmap="YlGnBu")
    plt.xlabel('True label',size = 14)
    plt.ylabel('Predicted label', size = 14)
    plt.xticks(np.arange(2)+0.8, Labname, size = 12)
    plt.yticks(np.arange(2)+0.4, Labname, size = 12)
    plt.savefig('result.png')
    plt.show()

    #----------------------------------第七 验证算法--------------------------
    ## 使用tok对验证数据集重新预处理，并使用训练好的模型进行预测
    val_seq = tok.texts_to_sequences(val_df.cutword)
    ## 将每个序列调整为相同的长度
    val_seq_mat = sequence.pad_sequences(val_seq,maxlen=max_len)
    ## 对验证集进行预测
    val_pre = model.predict(val_seq_mat)
    print(metrics.classification_report(np.argmax(val_y,axis=1),np.argmax(val_pre,axis=1)))
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)
