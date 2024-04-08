# encoding:utf-8
# By: Eastmount 2024-03-29
# keras-contrib=2.0.8  Keras=2.3.1  tensorflow=2.2.0  tensorflow-gpu=2.2.0  bert4keras=0.11.5
import re
import os
import csv
import sys
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import LSTM, GRU, Activation, Dense, Dropout, Input, Embedding, Permute
from keras.layers import Convolution1D, MaxPool1D, Flatten, TimeDistributed, Masking
from keras.optimizers import RMSprop
from keras.layers import Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.models import Sequential
from keras.layers.merge import concatenate
from keras import backend as K
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

#------------------------------------------------------------------------
#第一步 数据预处理
#------------------------------------------------------------------------
train_data_path = "data/train.csv"
test_data_path = "data/test.csv"
val_data_path = "data/val.csv"
char_vocab_path = "char_vocabs_.txt"   #字典文件（防止多次写入仅读首次生成文件）
special_words = ['<PAD>', '<UNK>']     #特殊词表示
final_words = []                       #统计词典（不重复出现）
final_labels = []                      #统计标记（不重复出现）

#BIO标记的标签 字母O初始标记为0
label2idx = {'O': 0,
             'S-LOC': 1, 'B-LOC': 2,  'I-LOC': 3,  'E-LOC': 4,
             'S-PER': 5, 'B-PER': 6,  'I-PER': 7,  'E-PER': 8,
             'S-TIM': 9, 'B-TIM': 10, 'E-TIM': 11, 'I-TIM': 12
             }
print(label2idx)
#{'S-LOC': 0, 'B-PER': 1, 'I-PER': 2, ...., 'I-TIM': 11, 'I-LOC': 12}

#索引和BIO标签对应
idx2label = {idx: label for label, idx in label2idx.items()}
print(idx2label)
#{0: 'S-LOC', 1: 'B-PER', 2: 'I-PER', ...., 11: 'I-TIM', 12: 'I-LOC'}

#读取字符词典文件
with open(char_vocab_path, "r", encoding="utf8") as fo:
    char_vocabs = [line.strip() for line in fo]
char_vocabs = special_words + char_vocabs
print(char_vocabs)
#['<PAD>', '<UNK>', '晉', '樂', '王', '鮒', '曰', '：', '小', '旻', ...]

# 字符和索引编号对应
idx2vocab = {idx: char for idx, char in enumerate(char_vocabs)}
vocab2idx = {char: idx for idx, char in idx2vocab.items()}
print(idx2vocab)
#{0: '<PAD>', 1: '<UNK>', 2: '晉', 3: '樂', ...}
print(vocab2idx)
#{'<PAD>': 0, '<UNK>': 1, '晉': 2, '樂': 3, ...}

#------------------------------------------------------------------------
#第二步 读取数据
#------------------------------------------------------------------------
def read_corpus(corpus_path, vocab2idx, label2idx):
    datas, labels = [], []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        line = line.strip()
        #print(line)
        if line != '':          #断句
            value = line.split(",")
            word,label = value[0],value[4]
            #汉字及标签逐一添加列表  ['晉', '樂'] ['S-LOC', 'B-PER']
            sent_.append(word)
            tag_.append(label)
            """
            print(sent_) #['晉', '樂', '王', '鮒', '曰', '：']
            print(tag_)  #['S-LOC', 'B-PER', 'I-PER', 'E-PER', 'O', 'O']
            """
        else:                   #vocab2idx[0] => <PAD>
            sent_ids = [vocab2idx[char] if char in vocab2idx else vocab2idx['<UNK>'] for char in sent_]
            tag_ids = [label2idx[label] if label in label2idx else 0 for label in tag_]
            datas.append(sent_ids) #按句插入列表
            labels.append(tag_ids)
            sent_, tag_ = [], []
    return datas, labels

#原始数据
train_datas_, train_labels_ = read_corpus(train_data_path, vocab2idx, label2idx)
test_datas_, test_labels_ = read_corpus(test_data_path, vocab2idx, label2idx)
val_datas_, val_labels_ = read_corpus(val_data_path, vocab2idx, label2idx)

#输出测试结果 (第五句语料)
print(len(train_datas_),len(train_labels_),len(test_datas_),
      len(test_labels_),len(val_datas_),len(val_labels_))
print(train_datas_[5])
print([idx2vocab[idx] for idx in train_datas_[5]])
print(train_labels_[5])
print([idx2label[idx] for idx in train_labels_[5]])

#------------------------------------------------------------------------
#第三步 数据填充 one-hot编码
#------------------------------------------------------------------------
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
#(15362, 100) (1919, 100)

#encoder one-hot
train_labels = keras.utils.to_categorical(train_labels, CLASS_NUMS)
test_labels = keras.utils.to_categorical(test_labels, CLASS_NUMS)
print('trainlabels shape:', train_labels.shape)
print('testlabels shape:', test_labels.shape)
#(15362, 100, 13) (1919, 100, 13)

#------------------------------------------------------------------------
#第四步 建立Attention机制
#------------------------------------------------------------------------
K.clear_session()
SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = concatenate([inputs, a_probs])
    return output_attention_mul

#------------------------------------------------------------------------
#第五步 构建ATT+CNN-BiLSTM+CRF模型
#------------------------------------------------------------------------
EPOCHS = 2
EMBED_DIM = 128
HIDDEN_SIZE = 64
MAX_LEN = 100
VOCAB_SIZE = len(vocab2idx)
CLASS_NUMS = len(label2idx)
print(VOCAB_SIZE, CLASS_NUMS) #3319 13

#模型构建
inputs = Input(shape=(MAX_LEN,), dtype='int32')
x = Masking(mask_value=0)(inputs)
x = Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=False)(x) #修改掩码False

#CNN
cnn1 = Convolution1D(64, 3, padding='same', strides = 1, activation='relu')(x)
cnn1 = MaxPool1D(pool_size=1)(cnn1)
cnn2 = Convolution1D(64, 4, padding='same', strides = 1, activation='relu')(x)
cnn2 = MaxPool1D(pool_size=1)(cnn2)
cnn3 = Convolution1D(64, 5, padding='same', strides = 1, activation='relu')(x)
cnn3 = MaxPool1D(pool_size=1)(cnn3)
cnn = concatenate([cnn1,cnn2,cnn3], axis=-1)
print(cnn.shape)   #(None, 100, 384)

#BiLSTM
bilstm = Bidirectional(LSTM(64, return_sequences=True))(cnn) #参数保持维度3 
layer = Dense(64, activation='relu')(bilstm)
layer = Dropout(0.3)(layer)
print(layer.shape) #(None, 100, 64)

#注意力
attention_mul = attention_3d_block(layer) #(None, 100, 128)
print(attention_mul.shape)

x = TimeDistributed(Dense(CLASS_NUMS))(attention_mul)
print(x.shape)     #(None, 3, 13)

outputs = CRF(CLASS_NUMS)(x)
print(outputs.shape)     #(None, 100, 13)
print(inputs.shape)      #(None, 100)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

#------------------------------------------------------------------------
#第六步 模型训练和预测
#------------------------------------------------------------------------
flag = "train"
if flag=="train":
    #模型训练
    model.compile(loss=crf_loss, optimizer='adam', metrics=[crf_viterbi_accuracy])
    model.fit(train_datas, train_labels, epochs=EPOCHS, verbose=1, validation_split=0.1)
    score = model.evaluate(test_datas, test_labels, batch_size=256)
    print(model.metrics_names)
    print(score)
    model.save("att_cnn_crf_bilstm_ner_model.h5")
elif flag=="test":
    #训练模型
    char_vocab_path = "char_vocabs_.txt"                #字典文件
    model_path = "att_cnn_crf_bilstm_ner_model.h5"      #模型文件
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
    print("最终结果大小:", len(final_y),len(final_z)) #191900 191900
    
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
    
    #文件存储
    fw = open("Final_ATT_CNN_BiLSTM_CRF_Result.csv", "w", encoding="utf8", newline='')
    fwrite = csv.writer(fw)
    fwrite.writerow(['pre_label','real_label', 'word'])
    n = 0
    while n<len(final_y):
        fwrite.writerow([final_y[n],final_z[n],final_word[n]])
        n += 1
    fw.close()

   
