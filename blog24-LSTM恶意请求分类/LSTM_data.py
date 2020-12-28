# coding=utf-8
# By：Eastmount CSDN 2020-11-15
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.models import load_model
from load_pj import classification_pj 
import time

start = time.clock()

#---------------------------------------第一步 数据读取------------------------------------
#读取测数据集
train_df = pd.read_csv("all_data_url_random_fenci_train.csv")
val_df = pd.read_csv("all_data_url_random_fenci_val.csv")
test_df = pd.read_csv("all_data_url_random_fenci_test.csv")
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
print(train_y[:10])  
print(len(train_y))

val_y = le.transform(val_y).reshape(-1,1)
test_y = le.transform(test_y).reshape(-1,1)

## 对数据集的标签数据进行one-hot编码
ohe = OneHotEncoder()
train_y = ohe.fit_transform(train_y).toarray()
val_y = ohe.transform(val_y).toarray()
test_y = ohe.transform(test_y).toarray()
print("OneHotEncoder:")
print(train_y[:10])

#-------------------------------第三步 使用Tokenizer对词组进行编码-------------------------------
#使用Tokenizer对词组进行编码
#当我们创建了一个Tokenizer对象后，使用该对象的fit_on_texts()函数，以空格去识别每个词
#可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
max_words = 5000
max_len = 600
tok = Tokenizer(num_words=max_words)  #使用的最大词语数为5000
tok.fit_on_texts(train_df.fenci)
print(tok)

#保存训练好的Tokenizer和导入
with open('tok.pickle', 'wb') as handle:
    pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
# loading
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
#对每个词编码之后，每句语料中的每个词就可以用对应的编码表示，即每条语料可以转变成一个向量了
train_seq = tok.texts_to_sequences(train_df.fenci)
val_seq = tok.texts_to_sequences(val_df.fenci)
test_seq = tok.texts_to_sequences(test_df.fenci)

#将每个序列调整为相同的长度
train_seq_mat = sequence.pad_sequences(train_seq,maxlen=max_len)
val_seq_mat = sequence.pad_sequences(val_seq,maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq,maxlen=max_len)

print(train_seq_mat.shape)  #(10000, 600)
print(val_seq_mat.shape)    #(5000, 600)
print(test_seq_mat.shape)   #(5000, 600)
print(train_seq_mat[:2])

#-------------------------------第四步 建立LSTM模型并训练-------------------------------
## 定义LSTM模型
inputs = Input(name='inputs',shape=[max_len])
## Embedding(词汇表大小,batch大小,每个新闻的词长)
layer = Embedding(max_words+1, 128, input_length=max_len)(inputs)
layer = LSTM(128)(layer)
layer = Dense(128, activation="relu", name="FC1")(layer)
layer = Dropout(0.3)(layer)
layer = Dense(2, activation="softmax", name="FC2")(layer)
model = Model(inputs=inputs, outputs=layer)
model.summary()
model.compile(loss="categorical_crossentropy",
              optimizer=RMSprop(),
              metrics=["accuracy"])

# 增加判断 防止再次训练
flag = "test"
if flag == "train":
    print("模型训练")
    #模型训练
    model_fit = model.fit(train_seq_mat, train_y, batch_size=128, epochs=10,
                          validation_data=(val_seq_mat,val_y),
                          callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)]   #当val-loss不再提升时停止训练
                         )
    
    #保存模型
    model.save('my_model.h5')  
    del model  # deletes the existing model
    
    #计算时间
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)

else:
    print("模型预测")
    # 导入已经训练好的模型
    model = load_model('my_model.h5')
    
    #--------------------------------------第五步 预测及评估--------------------------------
    #对测试集进行预测
    test_pre = model.predict(test_seq_mat)
    
    #评价预测效果，计算混淆矩阵 参数顺序
    confm = metrics.confusion_matrix(np.argmax(test_y,axis=1),np.argmax(test_pre,axis=1))
    print(confm)
    #混淆矩阵可视化
    Labname = ['正常', '异常']
    
    print(metrics.classification_report(np.argmax(test_y,axis=1),np.argmax(test_pre,axis=1)))
    classification_pj(np.argmax(test_pre,axis=1),np.argmax(test_y,axis=1))

    plt.figure(figsize=(8,8))
    sns.heatmap(confm.T, square=True, annot=True,
                fmt='d', cbar=False, linewidths=.6,
                cmap="YlGnBu")
    plt.xlabel('True label',size = 14)
    plt.ylabel('Predicted label', size = 14)
    plt.xticks(np.arange(2)+0.8, Labname, size = 12)
    plt.yticks(np.arange(2)+0.4, Labname, size = 12)
    plt.show()

    #--------------------------------------第六步 验证算法--------------------------------
    #使用tok对验证数据集重新预处理，并使用训练好的模型进行预测
    val_seq = tok.texts_to_sequences(val_df.fenci)
    
    #将每个序列调整为相同的长度
    val_seq_mat = sequence.pad_sequences(val_seq,maxlen=max_len)
    
    #对验证集进行预测
    val_pre = model.predict(val_seq_mat)
    print(metrics.classification_report(np.argmax(val_y,axis=1),np.argmax(val_pre,axis=1)))
    classification_pj(np.argmax(val_pre,axis=1),np.argmax(val_y,axis=1))
    
    #计算时间
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)
