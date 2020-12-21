# -*- coding:utf-8 -*-
import csv
import numpy as np
import jieba
import jieba.analyse
import jieba.posseg as pseg
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras import models
from keras import layers
from keras import Input
from gensim.models import word2vec
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Flatten, Dense, Dropout, Conv1D, MaxPool1D, Embedding

#----------------------------------第一步 数据预处理--------------------------------
file = "data.csv"

# 获取停用词
def stopwordslist(): #加载停用词表
    stopwords = [line.strip() for line in open('stop_words.txt', encoding="UTF-8").readlines()]
    return stopwords

# 去除停用词
def deleteStop(sentence):
    stopwords = stopwordslist()
    outstr = ""
    for i in sentence:
        # print(i)
        if i not in stopwords and i!="\n":
            outstr += i
    return outstr

# 中文分词
Mat = []
with open(file, "r", encoding="UTF-8") as f:
    # 使用csv.DictReader读取文件中的信息
    reader = csv.DictReader(f)
    labels = []
    contents = []
    for row in reader:
        # 数据元素获取
        if row['label'] == '好评':
            res = 0
        else:
            res = 1
        labels.append(res)

        # 中文分词
        content = row['content']
        #print(content)
        seglist = jieba.cut(content,cut_all=False)  #精确模式
        #print(seglist)
        
        # 去停用词
        stc = deleteStop(seglist) #注意此时句子无空格
        # 空格拼接
        seg_list = jieba.cut(stc,cut_all=False)
        output = ' '.join(list(seg_list))
        #print(output)
        contents.append(output)
        
        # 词性标注
        res = pseg.cut(stc)
        seten = []
        for word,flag in res:
            if flag not in ['nr','ns','nt','mz','m','f','ul','l','r','t']:
                #print(word,flag)
                seten.append(word)
        Mat.append(seten)

print(labels[:5])
print(contents[:5])
print(Mat[:5])

#----------------------------------第二步 特征编号--------------------------------
# fit_on_texts函数可以将输入的文本每个词编号 编号根据词频(词频越大编号越小)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(Mat)
vocab = tokenizer.word_index  #停用词已过滤,获取每个词的编号
print(vocab)

# 使用 train_test_split 分割 X y 列表
X_train, X_test, y_train, y_test = train_test_split(Mat, labels, test_size=0.3, random_state=1)
print(X_train[:5])
print(y_train[:5])

#----------------------------------第三步 词向量构建--------------------------------
# Word2Vec训练
maxLen = 100                 #词序列最大长度
num_features = 100           #设置词语向量维度
min_word_count = 3           #保证被考虑词语的最低频度
num_workers = 4              #设置并行化训练使用CPU计算核心数量
context = 4                  #设置词语上下文窗口大小

# 设置模型
model = word2vec.Word2Vec(Mat, workers=num_workers, size=num_features,
                          min_count=min_word_count,window=context)
# 强制单位归一化
model.init_sims(replace=True)
# 输入一个路径保存训练模型 其中./data/model目录事先存在
model.save("CNNw2vModel")
model.wv.save_word2vec_format("CNNVector",binary=False)
print(model)
# 加载模型 如果word2vec已训练好直接用下面语句
w2v_model = word2vec.Word2Vec.load("CNNw2vModel")

# 特征编号(不足的前面补0)
trainID = tokenizer.texts_to_sequences(X_train)
print(trainID)
testID = tokenizer.texts_to_sequences(X_test)
print(testID)
# 该方法会让CNN训练的长度统一
trainSeq = pad_sequences(trainID, maxlen=maxLen)
print(trainSeq)
testSeq = pad_sequences(testID, maxlen=maxLen)
print(testSeq)

# 标签独热编码 转换为one-hot编码
trainCate = to_categorical(y_train, num_classes=2) #二分类问题
print(trainCate)
testCate = to_categorical(y_test, num_classes=2)  #二分类问题
print(testCate)

#----------------------------------第四步 CNN构建--------------------------------
# 利用训练后的Word2vec自定义Embedding的训练矩阵 每行代表一个词(结合独热编码和矩阵乘法理解)
embedding_matrix = np.zeros((len(vocab)+1, 100)) #从0开始计数 加1对应之前特征词
for word, i in vocab.items():
    try:
        #提取词向量并放置训练矩阵
        embedding_vector = w2v_model[str(word)]
        embedding_matrix[i] = embedding_vector
    except KeyError: #单词未找到跳过
        continue

# 训练模型
main_input = Input(shape=(maxLen,), dtype='float64')
# 词嵌入 使用预训练Word2Vec的词向量 自定义权重矩阵 100是输出词向量维度
embedder = Embedding(len(vocab)+1, 100, input_length=maxLen, 
                     weights=[embedding_matrix], trainable=False) #不再训练
# 建立模型
model = Sequential()
model.add(embedder)  #构建Embedding层
model.add(Conv1D(256, 3, padding='same', activation='relu')) #卷积层步幅3
model.add(MaxPool1D(maxLen-5, 3, padding='same'))            #池化层
model.add(Conv1D(32, 3, padding='same', activation='relu'))  #卷积层
model.add(Flatten())                                         #拉直化
model.add(Dropout(0.3))                                      #防止过拟合 30%不训练
model.add(Dense(256, activation='relu'))                     #全连接层
model.add(Dropout(0.2))                                      #防止过拟合
model.add(Dense(units=2, activation='softmax'))              #输出层 

# 模型可视化
model.summary()

# 激活神经网络 
model.compile(optimizer = 'adam',                    #优化器
              loss = 'categorical_crossentropy',     #损失
              metrics = ['accuracy']                 #计算误差或准确率
              )

#训练(训练数据、训练类标、batch—size每次256条训练、epochs、随机选择、验证集20%)
history = model.fit(trainSeq, trainCate, batch_size=256, 
                    epochs=6, validation_split=0.2)
model.save("TextCNN")

#----------------------------------第五步 预测模型--------------------------------
# 预测与评估
mainModel = load_model("TextCNN")
result = mainModel.predict(testSeq) #测试样本
print(result)
print(np.argmax(result,axis=1))
score = mainModel.evaluate(testSeq,
                           testCate,
                           batch_size=32)
print(score)

#----------------------------------第五步 可视化--------------------------------
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Valid'], loc='upper left')


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Valid'], loc='upper left')
plt.show()