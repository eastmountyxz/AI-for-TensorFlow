#encoding:utf-8
#By:Eastmount CSDN
import re
import os
import csv
import numpy as np
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import Model
from keras.layers import Masking, Embedding, Bidirectional, LSTM, Dense
from keras.layers import Input, TimeDistributed, Activation
from keras.models import load_model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras import backend as K
from sklearn import metrics

#------------------------------------------------------------------------
#第一步 数据预处理
#------------------------------------------------------------------------
train_data_path = "dataset-train.txt"  #训练数据
test_data_path = "dataset-test.txt"    #测试数据
val_data_path = "dataset-val.txt"      #验证数据
char_vocab_path = "char_vocabs.txt"    #字典文件

special_words = ['<PAD>', '<UNK>']     #特殊词表示

#BIO标记的标签
label2idx = {"O": 0, "B-AG": 1, "B-AV": 2, "B-RL": 3,
             "B-AI":4, "B-AM": 5, "B-SI": 6, "B-OS": 7 }

# 索引和BIO标签对应
idx2label = {idx: label for label, idx in label2idx.items()}
print(idx2label)

# 读取字符词典文件
with open(char_vocab_path, "r", encoding="utf8") as fo:
    char_vocabs = [line.strip() for line in fo]
char_vocabs = special_words + char_vocabs
print(char_vocabs)
print("--------------------------------------------\n\n")

# 字符和索引编号对应 {'<PAD>': 0, '<UNK>': 1, 'APT-C-36': 2, ...}
idx2vocab = {idx: char for idx, char in enumerate(char_vocabs)}
vocab2idx = {char: idx for idx, char in idx2vocab.items()}
print(idx2vocab)
print("--------------------------------------------\n\n")
print(vocab2idx)
print("--------------------------------------------\n\n")

#------------------------------------------------------------------------
#第二步 读取训练语料
#------------------------------------------------------------------------
def read_corpus(corpus_path, vocab2idx, label2idx):
    datas, labels = [], []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':        #断句
            line = line.strip()
            [char, label] = line.split()
            sent_.append(char)
            tag_.append(label)
        else:
            #print(line)
            #vocab2idx[0] => <PAD>
            sent_ids = [vocab2idx[char] if char in vocab2idx else vocab2idx['<UNK>'] for char in sent_]
            tag_ids = [label2idx[label] if label in label2idx else 0 for label in tag_]
            datas.append(sent_ids)
            labels.append(tag_ids)
            sent_, tag_ = [], []
    return datas, labels

#原始数据
train_datas_, train_labels_ = read_corpus(train_data_path, vocab2idx, label2idx)
test_datas_, test_labels_ = read_corpus(test_data_path, vocab2idx, label2idx)

#输出测试结果 1639 1639 923 923
print(len(train_datas_), len(train_labels_), len(test_datas_), len(test_labels_))
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

# padding data
print('padding sequences')
train_datas = sequence.pad_sequences(train_datas_, maxlen=MAX_LEN)
train_labels = sequence.pad_sequences(train_labels_, maxlen=MAX_LEN)

test_datas = sequence.pad_sequences(test_datas_, maxlen=MAX_LEN)
test_labels = sequence.pad_sequences(test_labels_, maxlen=MAX_LEN)
print('x_train shape:', train_datas.shape)
print('x_test shape:', test_datas.shape)
# (1639, 100) (923, 100)

# encoder one-hot
train_labels = keras.utils.to_categorical(train_labels, CLASS_NUMS)
test_labels = keras.utils.to_categorical(test_labels, CLASS_NUMS)
print('trainlabels shape:', train_labels.shape)
print('testlabels shape:', test_labels.shape)
# (1639, 100, 8) (923, 100, 8)

#------------------------------------------------------------------------
#第四步 构建CRF模型
#------------------------------------------------------------------------
EPOCHS = 20
BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_SIZE = 64
MAX_LEN = 100
VOCAB_SIZE = len(vocab2idx)
CLASS_NUMS = len(label2idx)
K.clear_session()
print(VOCAB_SIZE, CLASS_NUMS, '\n') #3860 8

#模型构建 CRF
inputs = Input(shape=(MAX_LEN,), dtype='int32')
x = Masking(mask_value=0)(inputs)
x = Embedding(VOCAB_SIZE, 32, mask_zero=False)(x)
x = TimeDistributed(Dense(CLASS_NUMS))(x)
outputs = CRF(CLASS_NUMS)(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()

flag = "test"
if flag=="train":
    #模型训练
    model.compile(loss=crf_loss, optimizer='adam', metrics=[crf_viterbi_accuracy])
    model.fit(train_datas, train_labels, epochs=EPOCHS, verbose=1, validation_split=0.1)
    score = model.evaluate(test_datas, test_labels, batch_size=BATCH_SIZE)
    print(model.metrics_names)
    print(score)
    model.save("ch_ner_model.h5")
else:
    #------------------------------------------------------------------------
    #第五步 训练模型
    #------------------------------------------------------------------------
    char_vocab_path = "char_vocabs.txt"   #字典文件
    model_path = "ch_ner_model.h5"        #模型文件
    ner_labels = {"O": 0, "B-AG": 1, "B-AV": 2, "B-RL": 3,
                  "B-AI":4, "B-AM": 5, "B-SI": 6, "B-OS": 7 }
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
        #print(z)
        for idx in z:    
            final_z.append(idx2label[idx])
        #print("真实结果:", [idx2label[idx] for idx in z])
        word = word_labels[k]
        #print(word)
n         for idx in word:
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
    print(y_pred.shape)
    print(len(test_datas_), len(test_labels_))
    print("预测单词:", [idx2vocab[idx] for idx in test_datas_[0]])
    print("真实结果:", [idx2label[idx] for idx in test_labels_[0]])

    #文件存储
    fw = open("Final_CRF_Result.csv", "w", encoding="utf8", newline='')
    fwrite = csv.writer(fw)
    fwrite.writerow(['pre_label','real_label', 'word'])
    n = 0
    while n<len(final_y):
        fwrite.writerow([final_y[n],final_z[n],final_word[n]])
        n += 1
    fw.close()
