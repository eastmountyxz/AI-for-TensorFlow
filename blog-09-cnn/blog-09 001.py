# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:48:50 2019
@author: xiuzhang Eastmount CSDN
"""
from random import choice
from gensim.models import Word2Vec

#----------------------------------数据集生成----------------------------------
# 定义初始数据集
ls_of_ls = [['Python', '大数据', '人工智能', 'Tensorflow'], 
            ['网络安全', 'Web渗透', 'SQLMAP', 'Burpsuite'],
            ['网站开发', 'Java', 'MySQL', 'HTML5']]

# 真实项目中为数据集中文分词(jieba.cut)后的词列表
ls_of_words = []  
for i in range(1000):
    ls = choice(ls_of_ls) #随机选择某行数据集
    ls_of_words.append([choice(ls) for _ in range(9, 15)])
print(ls_of_words)

#----------------------------------词向量训练----------------------------------
# 训练
model = Word2Vec(ls_of_words)
print(model)

#-------------------------------计算词语之间相似度------------------------------
print(model.wv.similar_by_word('Tensorflow'))
print(model.wv.similarity('Web渗透', 'SQLMAP'))

#-------------------------------词矩阵计算------------------------------
# 显示词
print("【显示词语】")
print(model.wv.index2word)
# 显示词向量矩阵
print("【词向量矩阵】")
vectors = model.wv.vectors
print(vectors)
print(vectors.shape)
# 显示四个词语最相关的相似度
print("【词向量相似度】")
for i in range(4):
    print(model.wv.similar_by_vector(vectors[i]))

#-------------------------------预测新词------------------------------
print("【预测新词】")
print(model.predict_output_word(['人工智能']))
total = sum(i[1] for i in model.predict_output_word(['人工智能']))
print('概率总和为%.2f' % total)
