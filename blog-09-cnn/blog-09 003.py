# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:48:50 2019
@author: xiuzhang Eastmount CSDN
"""
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import matplotlib
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

#----------------------------------加载语料----------------------------------
file_name = "data_fc.txt"
ls_of_words = []  
f = open(file_name, encoding='utf-8')
for lines in f.readlines(): 
    words = lines.strip().split(" ")
    #print(words)
    ls_of_words.append(words)
print(ls_of_words)

#----------------------------------词向量训练----------------------------------
# 训练 size词向量维数100 vocab单词200
model = Word2Vec(sentences=ls_of_words, size=100, window=10)
print(model)
# 提取词向量
vectors = [model[word] for word in model.wv.index2word]
print(len(model.wv.index2word))

#----------------------------------词向量聚类----------------------------------
# 基于KMeans聚类
labels = KMeans(n_clusters=4).fit(vectors).labels_
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


