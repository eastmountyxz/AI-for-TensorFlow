# -*- coding: utf-8 -*-
"""
@author: xiuzhang Eastmount 2022-05-04
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers import Convolution1D, MaxPool1D, Flatten
from keras.optimizers import RMSprop
from keras.layers import Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.models import Sequential
from keras.layers.merge import concatenate
import tensorflow as tf

#GPU加速: 指定每个GPU进程中使用显存的上限 0.9表示可以使用GPU 90%的资源进行训练 CuDNNLSTM比LSTM快
os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#-----------------------------第一步 读取数据---------------------------------
#变量定义
train_cat = []
test_cat = []
train_label = []
test_label = []
train_review = []
test_review = []

#读取数据
train_path = 'data/online_shopping_10_cats_words_train.csv'
test_path = 'data/online_shopping_10_cats_words_test.csv'
types = {0: '消极', 1: '积极'}
pd_train = pd.read_csv(train_path)
pd_test = pd.read_csv(test_path)
print('训练集数目（总体）：%d' % pd_train.shape[0])
print('测试集数目（总体）：%d' % pd_test.shape[0])

for line in range(len(pd_train)):
    dict_cat = pd_train['cat'][line]
    dict_label = pd_train['label'][line]
    dict_content = str(pd_train['review'][line])
    train_cat.append(dict_cat)
    train_label.append(dict_label)
    train_review.append(dict_content)
print(len(train_cat),len(train_label),len(train_review))
print(train_cat[:5])
print(train_label[:5])

for line in range(len(pd_test)):
    dict_cat = pd_test['cat'][line]
    dict_label = pd_test['label'][line]
    dict_content = str(pd_test['review'][line])
    test_cat.append(dict_cat)
    test_label.append(dict_label)
    test_review.append(dict_content)
print(len(test_cat),len(test_label),len(test_review),"\n")

#------------------------第二步 OneHotEncoder()编码---------------------------
le = LabelEncoder()
train_y = le.fit_transform(train_label).reshape(-1, 1)
test_y = le.transform(test_label).reshape(-1, 1)
val_y = le.transform(train_label[:10000]).reshape(-1, 1)
print("LabelEncoder:")
print(len(train_y),len(test_y))
print(train_y[:10])

#对数据集的标签数据进行one-hot编码
ohe = OneHotEncoder()
train_y = ohe.fit_transform(train_y).toarray()
val_y = ohe.transform(val_y).toarray()
test_y = ohe.transform(test_y).toarray()
print("OneHotEncoder:")
print(train_y[:10])

#-----------------------第三步 使用Tokenizer对词组进行编码-----------------------
#Tokenizer将输入文本中的每个词编号 词频越大编号越小
max_words = 1200
max_len = 600
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(train_review)
print(tok)

#保存训练好的Tokenizer和导入
with open('tok.pickle', 'wb') as handle:
    pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('tok.pickle', 'rb') as handle:
    tok = pickle.load(handle)

#word_index属性查看词对应的编码 word_counts属性查看词对应的频数
for ii, iterm in enumerate(tok.word_index.items()):
    if ii < 10:
        print(iterm)
    else:
        break
for ii, iterm in enumerate(tok.word_counts.items()):
    if ii < 10:
        print(iterm)
    else:
        break

#序列转换
data_train = train_review
data_val = train_review[:10000]
data_test = test_review
train_seq = tok.texts_to_sequences(data_train)
val_seq = tok.texts_to_sequences(data_val)
test_seq = tok.texts_to_sequences(data_test)

#长度对齐
train_seq_mat = sequence.pad_sequences(train_seq, maxlen=max_len)
val_seq_mat = sequence.pad_sequences(val_seq, maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)
print(train_seq_mat.shape)
print(val_seq_mat.shape)
print(test_seq_mat.shape)
print(train_seq_mat[:2])

#--------------------------第四步 建立CNN模型---------------------------------
num_labels = 2
inputs = Input(name='inputs', shape=[max_len], dtype='float64')
layer = Embedding(max_words+1, 128, input_length=max_len, trainable=False)(inputs)
cnn = Convolution1D(128, 4, padding='same', strides = 1, activation='relu')(layer)
cnn = MaxPool1D(pool_size=4)(cnn)
flat = Flatten()(cnn)
drop = Dropout(0.4)(flat)
main_output = Dense(num_labels, activation='softmax')(drop)
model = Model(inputs=inputs, outputs=main_output)
model.summary()
model.compile(loss="categorical_crossentropy",
              optimizer='adam',      # RMSprop()
              metrics=["accuracy"])

#----------------------------第五步 训练和测试--------------------------------
flag = "test"
if flag == "train":
    print("模型训练:")
    model_fit = model.fit(train_seq_mat, train_y, 
                          batch_size=64, epochs=12,
                          validation_data=(val_seq_mat, val_y),
                          callbacks=[EarlyStopping(monitor='val_loss', 
                                                   min_delta=0.0005)]
                          )
    model.save('cnn_model.h5')
    del model  #deletes the existing model
else:
    print("模型预测")
    model = load_model('cnn_model.h5') #导入已经训练的模型
    test_pre = model.predict(test_seq_mat)    
    confm = metrics.confusion_matrix(np.argmax(test_pre, axis=1),
                                     np.argmax(test_y, axis=1))
    print(confm)
    with open("CNN-pre-result.txt","w") as f: #结果保存
        for v in np.argmax(test_pre, axis=1):
            f.write(str(v)+"\n")
    #混淆矩阵可视化
    Labname = ['积极', '消极']
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    print(metrics.classification_report(np.argmax(test_pre, axis=1),
                                        np.argmax(test_y, axis=1),
                                        digits=4))
    plt.figure(figsize=(8, 8))
    sns.heatmap(confm.T, square=True, annot=True,
                fmt='d', cbar=False, linewidths=.6,
                cmap="YlGnBu")
    plt.xlabel('True label', size=14)
    plt.ylabel('Predicted label', size=14)
    plt.xticks(np.arange(2)+0.5, Labname, size=12)
    plt.yticks(np.arange(2)+0.5, Labname, size=12)
    plt.savefig('CNN-result.png')
    plt.show()