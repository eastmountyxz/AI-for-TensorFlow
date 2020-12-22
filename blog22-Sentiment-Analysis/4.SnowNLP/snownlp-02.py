# -*- coding: utf-8 -*-
from snownlp import SnowNLP
import codecs
import os
import pandas as pd

#获取情感分数
f = open('庆余年220.csv',encoding='utf8')
data = pd.read_csv(f)
sentimentslist = []
for i in data['review']:
    s = SnowNLP(i)
    print(s.sentiments)
    sentimentslist.append(s.sentiments)

#区间转换为[-0.5, 0.5]
result = []
i = 0
while i<len(sentimentslist):
    result.append(sentimentslist[i]-0.5)
    i = i + 1

#可视化画图
import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.arange(0, 220, 1), result, 'k-')
plt.xlabel('Number')
plt.ylabel('Sentiment')
plt.title('Analysis of Sentiments')
plt.show()
