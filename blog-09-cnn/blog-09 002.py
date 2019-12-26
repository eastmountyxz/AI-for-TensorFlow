# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:48:50 2019
@author: xiuzhang Eastmount CSDN
"""
from random import choice
from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN
import matplotlib
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

#----------------------------------数据集生成----------------------------------
# 定义初始数据集
ls_of_ls = [['Python', '大数据', '人工智能', 'Tensorflow'], 
            ['网络安全', 'Web渗透', 'SQLMAP', 'Burpsuite'],
            ['网站开发', 'Java', 'MySQL', 'HTML5']]

# 真实项目中为数据集中文分词(jieba.cut)后的词列表
ls_of_words = []  
for i in range(2000):
    ls = choice(ls_of_ls) #随机选择某行数据集
    ls_of_words.append([choice(ls) for _ in range(9, 15)])
print(ls_of_words)

#----------------------------------词向量训练----------------------------------
# 训练 size词向量维数12 window预测距离6
model = Word2Vec(ls_of_words, size=3, window=7)
print(model)
# 提取词向量
vectors = [model[word] for word in model.wv.index2word]

#----------------------------------词向量聚类----------------------------------
# 基于密度的DBSCAN聚类
labels = DBSCAN(eps=0.24, min_samples=3).fit(vectors).labels_
print(labels)

#----------------------------------可视化显示----------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']         # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 显示负号
fig = plt.figure()
ax = mplot3d.Axes3D(fig)                             # 创建3d坐标轴
colors = ['red', 'blue', 'green', 'black']

# 绘制散点图 词语 词向量 类标(颜色)
for word, vector, label in zip(model.wv.index2word, vectors, labels):
    ax.scatter(vector[0], vector[1], vector[2], c=colors[label], s=500, alpha=0.4)
    ax.text(vector[0], vector[1], vector[2], word, ha='center', va='center')
plt.show()


