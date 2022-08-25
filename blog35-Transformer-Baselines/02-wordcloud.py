# -*- coding: utf-8 -*-
"""
@author: xiuzhang Eastmount 2022-05-04
"""
import pandas as pd
import csv
from collections import Counter

#-----------------------------------------------------------------------------
#读取分词后特征词
cut_words = ""
all_words = ""
pd_all = pd.read_csv('data/online_shopping_10_cats_words.csv')
moods = {0: '积极', 1: '消极'}
print('数目（总体）：%d' % pd_all.shape[0])    
for line in range(len(pd_all)): #cat label review
    dict_cat = pd_all['cat'][line]
    dict_label = pd_all['label'][line]
    dict_review = pd_all['review'][line]
    if str(dict_label)=="1":  #积极
        cut_words = dict_review
        all_words += str(cut_words)
        #print(cut_words)

#-----------------------------------------------------------------------------
#词频统计
all_words = all_words.split()
c = Counter()
for x in all_words:
    if len(x)>1 and x != '\r\n':
        c[x] += 1
print('\n词频统计结果：')
for (k,v) in c.most_common(10):
    print("%s:%d"%(k,v))
    
#存储数据
name ="data/data-word-count-train-happy.csv"
fw = open(name, 'w', encoding='utf-8')
i = 1
for (k,v) in c.most_common(len(c)):
    fw.write(str(i)+','+str(k)+','+str(v)+'\n')
    i = i + 1
else:
    print("Over write file!")
    fw.close()

#-----------------------------------------------------------------------------
#词云分析
from pyecharts import options as opts
from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType

#生成数据 word = [('A',10), ('B',9), ('C',8)] 列表+Tuple
words = []
for (k,v) in c.most_common(120):
    words.append((k,v))

#渲染图
def wordcloud_base() -> WordCloud:
    c = (
        WordCloud()
        .add("", words, word_size_range=[20, 40], shape='diamond') #shape=SymbolType.ROUND_RECT
        .set_global_opts(title_opts=opts.TitleOpts(title='词云图'))
    )
    return c
wordcloud_base().render('电商评论积极情感分析词云图.html')

