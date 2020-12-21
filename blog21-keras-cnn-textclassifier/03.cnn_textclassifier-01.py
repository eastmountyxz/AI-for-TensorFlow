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
                print(word,flag)
                seten.append(word)
        Mat.append(seten)
        break

print(labels[:5])
print(contents[:5])
print(Mat[:5])

#----------------------------------第二步 数据预处理--------------------------------
# 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer()

# 该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()

#第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(contents))
for n in tfidf[:5]:
    print(n)
#tfidf = tfidf.astype(np.float32)
print(type(tfidf))

# 获取词袋模型中的所有词语  
word = vectorizer.get_feature_names()
for n in word[:5]:
    print(n)
print("单词数量:", len(word)) 

# 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
X = tfidf.toarray()
print(X.shape)

# 使用 train_test_split 分割 X y 列表
# X_train矩阵的数目对应 y_train列表的数目(一一对应)  -->> 用来训练模型
# X_test矩阵的数目对应 	 (一一对应) -->> 用来测试模型的准确性
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=1)
