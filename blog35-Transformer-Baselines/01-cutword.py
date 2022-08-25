# -*- coding: utf-8 -*-
"""
@author: xiuzhang 2022-05-04
"""
import pandas as pd
import jieba
import csv
from collections import Counter

#-----------------------------------------------------------------------------
#样本数量统计
pd_all = pd.read_csv('data/online_shopping_10_cats.csv')
moods = {0: '消极', 1: '积极'}

print('数目（总体）：%d' % pd_all.shape[0])
for label, mood in moods.items(): 
    print('数目（{}）：{}'.format(mood, pd_all[pd_all.label==label].shape[0]))


#-----------------------------------------------------------------------------
#中文分词和停用词过滤
cut_words = ""
all_words = ""
stopwords = ["[", "]", "）", "（", ")", "(", "【", "】", "！", "，", "$",
             "·", "？", ".", "、", "-", "—", ":", "：", "《", "》", "=",
             "。", "…", "“", "?", "”", "~", " ", "－", "+", "\\", "‘",
             "～", "；", "’", "...", "..", "&", "#",  "....", ",", 
             "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
             "的", "和", "之", "了", "哦", "那", "一个",  ]

fw = open('data/online_shopping_10_cats_words.csv', 'w', encoding='utf-8')
for line in range(len(pd_all)): #cat label review
    dict_cat = pd_all['cat'][line]
    dict_label = pd_all['label'][line]
    dict_review = pd_all['review'][line]
    dict_review = str(dict_review)
    #print(dict_cat, dict_label, dict_review)

    #jieba分词并过滤停用词
    cut_words = ""
    data = dict_review.strip('\n')
    data = data.replace(",", "")    #一定要过滤符号 ","否则多列
    seg_list = jieba.cut(data, cut_all=False)
    for seg in seg_list:
        if seg not in stopwords:
            cut_words += seg + " "
    all_words += cut_words
    #print(cut_words)
    
    fw.write(str(dict_cat)+","+str(dict_label)+","+str(cut_words)+"\n")
else:
    fw.close()

#-----------------------------------------------------------------------------
#词频统计
all_words = all_words.split()
print(all_words)

c = Counter()
for x in all_words:
    if len(x)>1 and x != '\r\n':
        c[x] += 1
        
#输出词频最高的前10个词
print('\n词频统计结果：')
for (k,v) in c.most_common(10):
    print("%s:%d"%(k,v))

