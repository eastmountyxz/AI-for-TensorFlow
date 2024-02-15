#encoding:utf-8
# By: Eastmount WuShuai 2024-02-05
# 参考:https://github.com/huanghao128/zh-nlp-demo
import re
import os
import csv
import sys
from get_data import build_vocab #调取第一阶段函数

#------------------------------------------------------------------------
#第一步 数据预处理
#------------------------------------------------------------------------
train_data_path = "data/train.csv"
test_data_path = "data/test.csv"
val_data_path = "data/val.csv"
char_vocab_path = "char_vocabs.txt"   #字典文件（防止多次写入仅读首次生成文件）
special_words = ['<PAD>', '<UNK>']     #特殊词表示
final_words = []                       #统计词典（不重复出现）
final_labels = []                      #统计标记（不重复出现）

#BIO标记的标签 字母O初始标记为0
#label2idx = build_vocab()
label2idx = {'O': 0,
             'S-LOC': 1, 'B-LOC': 2,  'I-LOC': 3,  'E-LOC': 4,
             'S-PER': 5, 'B-PER': 6,  'I-PER': 7,  'E-PER': 8,
             'S-TIM': 9, 'B-TIM': 10, 'E-TIM': 11, 'I-TIM': 12
             }
print(label2idx)

#索引和BIO标签对应
idx2label = {idx: label for label, idx in label2idx.items()}
print(idx2label)

#读取字符词典文件
with open(char_vocab_path, "r") as fo:
    char_vocabs = [line.strip() for line in fo]
char_vocabs = special_words + char_vocabs
print(char_vocabs)

#字符和索引编号对应
idx2vocab = {idx: char for idx, char in enumerate(char_vocabs)}
vocab2idx = {char: idx for idx, char in idx2vocab.items()}
print(idx2vocab)
print(vocab2idx)

#------------------------------------------------------------------------
#第二步 数据读取
#------------------------------------------------------------------------
def read_corpus(corpus_path, vocab2idx, label2idx):
    datas, labels = [], []
    with open(corpus_path, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        sent_, tag_ = [], []
        for row in reader:
            word,label = row[0],row[1]
            if word!="" and label!="":   #断句
                sent_.append(word)
                tag_.append(label)
                """
                print(sent_) #['晉', '樂', '王', '鮒', '曰', '：']
                print(tag_)  #['S-LOC', 'B-PER', 'I-PER', 'E-PER', 'O', 'O']
                """
            else:                        #vocab2idx[0] => <PAD>
                sent_ids = [vocab2idx[char] if char in vocab2idx else vocab2idx['<UNK>'] for char in sent_]
                tag_ids = [label2idx[label] if label in label2idx else 0 for label in tag_]
                """
                print(sent_ids,tag_ids)
                for idx,idy in zip(sent_ids,tag_ids):
                    print(idx2vocab[idx],idx2label[idy])
                #[2, 3, 4, 5, 6, 7] [1, 6, 7, 8, 0, 0]
                #晉 S-LOC 樂 B-PER 王 I-PER 鮒 E-PER 曰 O ： O
                """
                datas.append(sent_ids) #按句插入列表
                labels.append(tag_ids)
                sent_, tag_ = [], []
    return datas, labels

#原始数据
train_datas_, train_labels_ = read_corpus(train_data_path, vocab2idx, label2idx)
test_datas_, test_labels_ = read_corpus(test_data_path, vocab2idx, label2idx)

#输出测试结果 (第五句语料)
print(len(train_datas_),len(train_labels_),len(test_datas_),len(test_labels_))
print(train_datas_[5])
print([idx2vocab[idx] for idx in train_datas_[5]])
print(train_labels_[5])
print([idx2label[idx] for idx in train_labels_[5]])

#------------------------------------------------------------------------
#第三步 数据填充 one-hot编码
#------------------------------------------------------------------------
import keras
from keras.preprocessing import sequence

MAX_LEN = 100
VOCAB_SIZE = len(vocab2idx)
CLASS_NUMS = len(label2idx)

#padding data
print('padding sequences')
train_datas = sequence.pad_sequences(train_datas_, maxlen=MAX_LEN)
train_labels = sequence.pad_sequences(train_labels_, maxlen=MAX_LEN)
test_datas = sequence.pad_sequences(test_datas_, maxlen=MAX_LEN)
test_labels = sequence.pad_sequences(test_labels_, maxlen=MAX_LEN)
print('x_train shape:', train_datas.shape)
print('x_test shape:', test_datas.shape)

#encoder one-hot
train_labels = keras.utils.to_categorical(train_labels, CLASS_NUMS)
test_labels = keras.utils.to_categorical(test_labels, CLASS_NUMS)
print('trainlabels shape:', train_labels.shape)
print('testlabels shape:', test_labels.shape)

#------------------------------------------------------------------------
#第四步 构建BiLSTM+CRF模型
# pip install git+https://www.github.com/keras-team/keras-contrib.git
# 安装过程详见文件夹截图
# ModuleNotFoundError: No module named ‘keras_contrib’
#------------------------------------------------------------------------
import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers import Masking, Embedding, Bidirectional, LSTM, \
     Dense, Input, TimeDistributed, Activation
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras import backend as K
from keras.models import load_model
from sklearn import metrics

EPOCHS = 2
EMBED_DIM = 128
HIDDEN_SIZE = 64
MAX_LEN = 100
VOCAB_SIZE = len(vocab2idx)
CLASS_NUMS = len(label2idx)
K.clear_session()
print(VOCAB_SIZE, CLASS_NUMS) #3319 13

#模型构建 BiLSTM-CRF
inputs = Input(shape=(MAX_LEN,), dtype='int32')
x = Masking(mask_value=0)(inputs)
x = Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=False)(x) #修改掩码False
x = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True))(x)
x = TimeDistributed(Dense(CLASS_NUMS))(x)
outputs = CRF(CLASS_NUMS)(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()

flag = "test"
if flag=="train":
    #模型训练
    model.compile(loss=crf_loss, optimizer='adam', metrics=[crf_viterbi_accuracy])
    model.fit(train_datas, train_labels, epochs=EPOCHS, verbose=1, validation_split=0.1)
    score = model.evaluate(test_datas, test_labels, batch_size=256)
    print(model.metrics_names)
    print(score)
    model.save("bilstm_ner_model.h5")
elif flag=="test":
    #训练模型
    char_vocab_path = "char_vocabs_.txt"      #字典文件
    model_path = "bilstm_ner_model.h5"        #模型文件
    ner_labels = label2idx
    special_words = ['<PAD>', '<UNK>']
    MAX_LEN = 100
    
    #预测结果
    model = load_model(model_path, custom_objects={'CRF': CRF}, compile=False)    
    y_pred = model.predict(test_datas)
    y_labels = np.argmax(y_pred, axis=2)         #取最大值
    z_labels = np.argmax(test_labels, axis=2)    #真实值
    word_labels = test_datas                     #真实值
    
    k = 0
    final_y = []       #预测结果对应的标签
    final_z = []       #真实结果对应的标签
    final_word = []    #对应的特征单词
    while k<len(y_labels):
        y = y_labels[k]
        for idx in y:
            final_y.append(idx2label[idx])
        #print("预测结果:", [idx2label[idx] for idx in y])
        
        z = z_labels[k]
        for idx in z:    
            final_z.append(idx2label[idx])
        #print("真实结果:", [idx2label[idx] for idx in z])
        
        word = word_labels[k]
        for idx in word:
            final_word.append(idx2vocab[idx])
        k += 1
    print("最终结果大小:", len(final_y),len(final_z))
    
    n = 0
    numError = 0
    numRight = 0
    while n<len(final_y):
        if final_y[n]!=final_z[n] and final_z[n]!='O':
            numError += 1
        if final_y[n]==final_z[n] and final_z[n]!='O':
            numRight += 1
        n += 1
    print("预测错误数量:", numError)
    print("预测正确数量:", numRight)
    print("Acc:", numRight*1.0/(numError+numRight))
    print(y_pred.shape, len(test_datas_), len(test_labels_))
    print("预测单词:", [idx2vocab[idx] for idx in test_datas_[5]])
    print("真实结果:", [idx2label[idx] for idx in test_labels_[5]])
    print("预测结果:", [idx2label[idx] for idx in y_labels[5]][-len(test_datas_[5]):])
    
