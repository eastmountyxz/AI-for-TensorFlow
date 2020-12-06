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
from tensorflow.python.keras.layers import Dense, GRU, Bidirectional
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

#中文分析
X, Y = [lcut(i[1]) for i in data], [i[0] for i in data]

#划分训练集和预测集
X_train, X_test, y_train, y_test = train_test_split(X, Y)
#print(X_train)
print(len(X_train), len(X_test))
print(len(y_train), len(y_test))
"""['三星', '刚刚', '更新', '了', '自家', '的', '可', '穿戴', '设备', 'APP']"""

#--------------------------------Word2Vec词向量-------------------------------
word2vec = Word2Vec(X_train, size=max_features, min_count=1) #最大特征 最低过滤频次1
print(word2vec)

#映射特征词
w2i = {w:i for i, w in enumerate(word2vec.wv.index2word)}
print("【显示词语】")
print(word2vec.wv.index2word)
print(w2i)
"""['小米', '三星', '是', '维生素', '蛋白质', '及', 'APP', '氨基酸',..."""
"""{'，': 0, '的': 1, '小米': 2, '、': 3, '华为': 4, ....}"""

#词向量计算
vectors = word2vec.wv.vectors
print("【词向量矩阵】")
print(vectors.shape)
print(vectors)

#自定义函数-获取词向量
def w2v(w):
    i = w2i.get(w)
    return vectors[i] if i else zeros(max_features)

#自定义函数-序列预处理
def pad(ls_of_words):
    a = [[w2v(i) for i in x] for x in ls_of_words]
    a = pad_sequences(a, maxlen, dtype='float')
    return a

#序列化处理 转换为词向量
X_train, X_test = pad(X_train), pad(X_test)

#--------------------------------建模与训练-------------------------------
model = Sequential()

#双向RNN
model.add(Bidirectional(GRU(units), input_shape=(maxlen, max_features)))  

#输出层 2分类
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
plt.title("RNN-Word2vec")
plt.show()
