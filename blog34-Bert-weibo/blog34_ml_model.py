# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:21:53 2021
@author: xiuzhang
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
#读取数据
train_path = 'data/weibo_3_moods_train.csv'
test_path = 'data/weibo_3_moods_test.csv'
types = {0: '喜悦', 1: '愤怒', 2: '哀伤'}
pd_train = pd.read_csv(train_path)
pd_test = pd.read_csv(test_path)
print('训练集数目（总体）：%d' % pd_train.shape[0])
print('测试集数目（总体）：%d' % pd_test.shape[0])

#中文分词
train_words = []
test_words = []
train_labels = []
test_labels = []
stopwords = ["[", "]", "）", "（", ")", "(", "【", "】", "！", "，", "$",
             "·", "？", ".", "、", "-", "—", ":", "：", "《", "》", "=",
             "。", "…", "“", "?", "”", "~", " ", "－", "+", "\\", "‘",
             "～", "；", "’", "...", "..", "&", "#",  "....", ",", 
             "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
             "的", "和", "之", "了", "哦", "那", "一个",  ]

for line in range(len(pd_train)):
    dict_label = pd_train['label'][line]
    dict_content = str(pd_train['content'][line]) #float=>str
    #print(dict_label,dict_content)
    cut_words = ""
    data = dict_content.strip("\n")
    data = data.replace(",", "")    #一定要过滤符号 ","否则多列
    seg_list = jieba.cut(data, cut_all=False)
    for seg in seg_list:
        if seg not in stopwords:
            cut_words += seg + " "
    #print(cut_words)
    label = -1
    if dict_label=="喜悦":
        label = 0
    elif dict_label=="愤怒":
        label = 1
    elif dict_label=="哀伤":
        label = 2
    else:
        label = -1
    train_labels.append(label)
    train_words.append(cut_words)
print(len(train_labels),len(train_words)) #209043 209043

for line in range(len(pd_test)):
    dict_label = pd_test['label'][line]
    dict_content = str(pd_test['content'][line])
    cut_words = ""
    data = dict_content.strip("\n")
    data = data.replace(",", "")
    seg_list = jieba.cut(data, cut_all=False)
    for seg in seg_list:
        if seg not in stopwords:
            cut_words += seg + " "
    label = -1
    if dict_label=="喜悦":
        label = 0
    elif dict_label=="愤怒":
        label = 1
    elif dict_label=="哀伤":
        label = 2
    else:
        label = -1
    test_labels.append(label)
    test_words.append(cut_words)
print(len(test_labels),len(test_words)) #97366 97366
print(test_labels[:5])                  #['喜悦', '喜悦', '愤怒', '哀伤', '喜悦']

#-----------------------------------------------------------------------------
#TFIDF计算
#将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer(min_df=100)   #MemoryError控制参数

#该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()

#第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(train_words+test_words))
for n in tfidf[:5]:
    print(n)
print(type(tfidf))

#获取词袋模型中的所有词语  
word = vectorizer.get_feature_names()
for n in word[:10]:
    print(n)
print("单词数量:", len(word))

#将tf-idf矩阵抽取 元素w[i][j]表示j词在i类文本中的tf-idf权重
X = coo_matrix(tfidf, dtype=np.float32).toarray()  #稀疏矩阵
print(X.shape)
print(X[:10])

X_train = X[:len(train_labels)]
X_test = X[len(train_labels):]
y_train = train_labels
y_test = test_labels
print(len(X_train),len(X_test),len(y_train),len(y_test))

#-----------------------------------------------------------------------------
#分类模型
clf = MultinomialNB()
#clf = svm.LinearSVC()
#clf = LogisticRegression(solver='liblinear')
#clf = RandomForestClassifier(n_estimators=10)
#clf = neighbors.KNeighborsClassifier(n_neighbors=7)
#clf = AdaBoostClassifier()
clf.fit(X_train, y_train)
print('模型的准确度:{}'.format(clf.score(X_test, y_test)))
pre = clf.predict(X_test)
print("分类")
print(len(pre), len(y_test))
print(classification_report(y_test, pre, digits=4))

