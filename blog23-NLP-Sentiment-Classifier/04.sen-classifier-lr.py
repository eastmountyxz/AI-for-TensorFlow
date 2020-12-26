# -*- coding:utf-8 -*-
import csv
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
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

#----------------------------------第一步 读取文件--------------------------------
with open('fenci_data.csv', 'r', encoding='UTF-8') as f:
    reader = csv.DictReader(f)
    labels = []
    contents = []
    for row in reader:
        labels.append(row['label']) #0-好评 1-差评
        contents.append(row['content'])

print(labels[:5])
print(contents[:5])

#----------------------------------第二步 数据预处理--------------------------------
#将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer(min_df=5)

#该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()

#第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(contents))
for n in tfidf[:5]:
    print(n)
print(type(tfidf))

# 获取词袋模型中的所有词语  
word = vectorizer.get_feature_names()
for n in word[:10]:
    print(n)
print("单词数量:", len(word))

#将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
#X = tfidf.toarray()
X = coo_matrix(tfidf, dtype=np.float32).toarray() #稀疏矩阵 注意float
print(X.shape)
print(X[:10])

#----------------------------------第三步 数据划分--------------------------------
#使用 train_test_split 分割 X y 列表
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    labels, 
                                                    test_size=0.3, 
                                                    random_state=1)

#--------------------------------第四步 机器学习分类--------------------------------
# 逻辑回归分类方法模型
LR = LogisticRegression(solver='liblinear')
LR.fit(X_train, y_train)
print('模型的准确度:{}'.format(LR.score(X_test, y_test)))
pre = LR.predict(X_test)
print("逻辑回归分类")
print(len(pre), len(y_test))
print(classification_report(y_test, pre))
print("\n")


