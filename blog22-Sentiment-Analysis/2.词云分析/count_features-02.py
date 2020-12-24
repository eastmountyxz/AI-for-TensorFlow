# coding: utf-8
import csv
import pandas as pd
import operator

#------------------------------------统计结果------------------------------------
#读取数据
f = open('Emotion_features.csv')
data = pd.read_csv(f)
print(data.head())

#统计结果
groupnum = data.groupby(['Emotion']).size()
print(groupnum)
print("")

#分组统计
for groupname,grouplist in data.groupby('Emotion'):
    print(groupname)
    print(grouplist)

#生成数据 word = [('A',10), ('B',9), ('C',8)] 列表+Tuple
i = 0
words = []
counts = []
while i<len(data):
    if data['Emotion'][i] in "disgust": #相等
        k = data['Word'][i]
        v = data['Num'][i]
        
        n = 0
        flag = 0
        while n<len(words):
            #如果两个单词相同则增加次数
            if words[n]==k:
                counts[n] = counts[n] + v
                flag = 1
                break
            n = n + 1
        #如果没有找到相同的特征词则添加
        if flag==0:
            words.append(k)
            counts.append(v)
    i = i + 1

#添加最终数组结果
result = []
k = 0
while k<len(words):
    result.append((words[k], int(counts[k]*5)))  #注意：因数据集较少,作者扩大5倍方便绘图
    k = k + 1
print(result)

#------------------------------------词云分析------------------------------------
from pyecharts import options as opts
from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType

# 渲染图
def wordcloud_base() -> WordCloud:
    c = (
        WordCloud()
        .add("", result, word_size_range=[5, 200], shape=SymbolType.ROUND_RECT)
        .set_global_opts(title_opts=opts.TitleOpts(title='情绪词云图'))
    )
    return c

# 生成图
wordcloud_base().render('情绪词云图.html')
