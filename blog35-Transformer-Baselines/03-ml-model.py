# -*- coding: utf-8 -*-
"""
@author: xiuzhang Eastmount 2022-05-04
"""
import jieba
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
#变量定义
train_cat = []
test_cat = []
train_label = []
test_label = []
train_review = []
test_review = []

#读取数据
train_path = 'data/online_shopping_10_cats_words_train.csv'
test_path = 'data/online_shopping_10_cats_words_test.csv'
types = {0: '消极', 1: '积极'}
pd_train = pd.read_csv(train_path)
pd_test = pd.read_csv(test_path)
print('训练集数目（总体）：%d' % pd_train.shape[0])
print('测试集数目（总体）：%d' % pd_test.shape[0])

for line in range(len(pd_train)):
    dict_cat = pd_train['cat'][line]
    dict_label = pd_train['label'][line]
    dict_content = str(pd_train['review'][line])
    train_cat.append(dict_cat)
    train_label.append(dict_label)
    train_review.append(dict_content)
print(len(train_cat),len(train_label),len(train_review))
print(train_cat[:5])
print(train_label[:5])

for line in range(len(pd_test)):
    dict_cat = pd_test['cat'][line]
    dict_label = pd_test['label'][line]
    dict_content = str(pd_test['review'][line])
    test_cat.append(dict_cat)
    test_label.append(dict_label)
    test_review.append(dict_content)
print(len(test_cat),len(test_label),len(test_review),"\n")

#-----------------------------------------------------------------------------
#TFIDF计算
#将文本中的词语转换为词频矩阵 矩阵元素a[i][j]表示词j在第i类文本下的词频
vectorizer = CountVectorizer(min_df=10)   #MemoryError控制参数

#统计每个词语的tf-idf权值
transformer = TfidfTransformer()

#第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(train_review+test_review))
for n in tfidf[:5]:
    print(n)
print(type(tfidf))

#获取词袋模型中的所有词语  
word = vectorizer.get_feature_names()
print("单词数量:", len(word))

#元素w[i][j]表示词j在第i类文本中的tf-idf权重
X = coo_matrix(tfidf, dtype=np.float32).toarray()  #稀疏矩阵
print(X.shape)
print(X[:10])

X_train = X[:len(train_label)]
X_test = X[len(train_label):]
y_train = train_label
y_test = test_label
print(len(X_train),len(X_test),len(y_train),len(y_test))

#-----------------------------------------------------------------------------
#分类模型
#clf = svm.LinearSVC()
#clf = LogisticRegression(solver='liblinear')
#clf = RandomForestClassifier(n_estimators=11)
#clf = DecisionTreeClassifier()
#clf = AdaBoostClassifier()
#clf = MultinomialNB()
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
print(clf)

clf.fit(X_train, y_train)
pre = clf.predict(X_test)
print('模型的准确度:{}'.format(clf.score(X_test, y_test)))
print(len(pre), len(y_test))
print(classification_report(y_test, pre, digits=4))
with open("KNN-pre-result.txt","w") as f:  #结果保存
    for v in pre:
        f.write(str(v)+"\n")

