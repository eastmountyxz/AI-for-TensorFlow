# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:21:53 2021
@author: xiuzhang
"""
import pandas as pd
import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

#-----------------------------------------------------------------------------
#读取分词后特征词
pd_all = pd.read_csv('weibo_3_fenci.csv')
moods = {0: '喜悦', 1: '愤怒', 2: '低落'}
print('微博数目（总体）：%d' % pd_all.shape[0])
for label, mood in moods.items(): 
    print('微博数目（{}）：{}'.format(mood,pd_all[pd_all.label==label].shape[0]))

labels = []
contents = []
for line in range(len(pd_all)):  #label review
    labels.append(pd_all['label'][line])
    contents.append(str(pd_all['review'][line]))
print(len(labels),len(contents))

#-----------------------------------------------------------------------------
#TFIDF计算
#将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer(min_df=100)   #MemoryError控制参数

#该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()

#第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(contents))
for n in tfidf[:5]:
    print(n)
print(type(tfidf))

#获取词袋模型中的所有词语  
word = vectorizer.get_feature_names()
for n in word[:10]:
    print(n)
print("单词数量:", len(word))  #74806 12884

#将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
X = coo_matrix(tfidf, dtype=np.float32).toarray() #稀疏矩阵 float
#X = tfidf.toarray()
print(X.shape)
print(X[:10])

#数据划分
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    labels, 
                                                    test_size=0.3, 
                                                    random_state=1)
print(len(X_train),len(X_test),len(y_train),len(y_test))

#-----------------------------------------------------------------------------
# 逻辑回归分类方法模型
#clf = LogisticRegression(solver='liblinear')
clf = RandomForestClassifier(n_estimators=10)
#clf = svm.LinearSVC()
#clf = MultinomialNB()
#clf = neighbors.KNeighborsClassifier(n_neighbors=7)
#clf = AdaBoostClassifier()
clf.fit(X_train, y_train)
print('模型的准确度:{}'.format(clf.score(X_test, y_test)))
pre = clf.predict(X_test)
print("分类")
print(len(pre), len(y_test))
print(classification_report(y_test, pre, digits=4))
print("\n")

