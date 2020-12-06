# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:10:20 2020
@author: Eastmount CSDN
"""
from jieba import lcut
from numpy import zeros
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, GRU, Embedding
from tensorflow.python.keras.callbacks import EarlyStopping

#-----------------------------------定义参数----------------------------------
max_features = 20                  #词向量维度
units = 30                         #RNN神经元数量
maxlen = 40                        #序列最大长度
epochs = 9                         #训练最大轮数
batch_size = 12                    #每批数据量大小
verbose = 1                        #训练过程展示
patience = 1                       #没有进步的训练轮数

callbacks = [EarlyStopping('val_acc', patience=patience)]

#--------------------------------载入数据及预处理-------------------------------
data = [
    [0, '小米粥是以小米作为主要食材熬制而成的粥，口味清淡，清香味，具有简单易制，健胃消食的特点'],
    [0, '煮粥时一定要先烧开水然后放入洗净后的小米'], 
    [0, '蛋白质及氨基酸、脂肪、维生素、矿物质'],
    [0, '小米是传统健康食品，可单独焖饭和熬粥'], 
    [0, '苹果，是水果中的一种'],
    [0, '粥的营养价值很高，富含矿物质和维生素，含钙量丰富，有助于代谢掉体内多余盐分'],
    [0, '鸡蛋有很高的营养价值，是优质蛋白质、B族维生素的良好来源，还能提供脂肪、维生素和矿物质'],
    [0, '这家超市的苹果都非常新鲜'], 
    [0, '在北方小米是主要食物之一，很多地区有晚餐吃小米粥的习俗'],
    [0, '小米营养价值高，营养全面均衡 ，主要含有碳水化合物'], 
    [0, '蛋白质及氨基酸、脂肪、维生素、盐分'],
    [1, '小米、三星、华为，作为安卓三大手机旗舰'], 
    [1, '别再管小米华为了！魅族手机再曝光：这次真的完美了'],
    [1, '苹果手机或将重陷2016年困境，但这次它无法再大幅提价了'], 
    [1, '三星想要继续压制华为，仅凭A70还不够'],
    [1, '三星手机屏占比将再创新高，超华为及苹果旗舰'], 
    [1, '华为P30、三星A70爆卖，斩获苏宁最佳手机营销奖'],
    [1, '雷军，用一张图告诉你：小米和三星的差距在哪里'], 
    [1, '小米米聊APP官方Linux版上线，适配深度系统'],
    [1, '三星刚刚更新了自家的可穿戴设备APP'], 
    [1, '华为、小米跨界并不可怕，可怕的打不破内心的“天花板”'],
]

#中文分词
X, Y = [' '.join(lcut(i[1])) for i in data], [i[0] for i in data]
print(X)
print(Y)
#['煮粥 时 一定 要 先烧 开水 然后 放入 洗净 后 的 小米', ...]

#--------------------------------------计算词频------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()

#计算个词语出现的次数
X_data = vectorizer.fit_transform(X)
print(X_data)

#获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
print('【查看单词】')
for w in word:
    print(w, end = " ")
else:
    print("\n")

#词频矩阵
print(X_data.toarray())

#将词频矩阵X统计成TF-IDF值
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X_data)

#查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
weight = tfidf.toarray()
print(weight)

#数据集划分
X_train, X_test, y_train, y_test = train_test_split(weight, Y)
print(X_train.shape, X_test.shape)
print(len(y_train), len(y_test))
#(15, 117) (6, 117) 15 6

#--------------------------------建模与训练-------------------------------
model = Sequential()

#构建Embedding层 128代表Embedding层的向量维度
model.add(Embedding(max_features, 128))

#构建LSTM层
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

#构建全连接层
#注意上面构建LSTM层时只会得到最后一个节点的输出,如果需要输出每个时间点的结果需将return_sequences=True
model.add(Dense(units=1, activation='sigmoid'))

#模型可视化
model.summary()

#激活神经网络 
model.compile(optimizer = 'rmsprop',              #RMSprop优化器
              loss = 'binary_crossentropy',       #二元交叉熵损失
              metrics = ['acc']                   #计算误差或准确率
              )

#训练
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
                    verbose=verbose, validation_data=(X_test, y_test))

#----------------------------------预测与可视化------------------------------
#预测
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print('test loss:', score[0])
print('test accuracy:', score[1])

#可视化
acc = history.history['acc']
val_acc = history.history['val_acc']

# 设置类标
plt.xlabel("Iterations")
plt.ylabel("Accuracy")

#绘图
plt.plot(range(epochs), acc, "bo-", linewidth=2, markersize=12, label="accuracy")
plt.plot(range(epochs), val_acc, "gs-", linewidth=2, markersize=12, label="val_accuracy")
plt.legend(loc="upper left")
plt.title("LSTM-TFIDF")
plt.show()