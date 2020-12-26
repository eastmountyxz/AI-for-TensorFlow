# -*- coding:utf-8 -*-
import csv
import pandas as pd
import numpy as np
import jieba
import jieba.analyse

#添加自定义词典和停用词典
jieba.load_userdict("user_dict.txt")
stop_list = pd.read_csv('stop_words.txt',
                        engine='python',
                        encoding='utf-8',
                        delimiter="\n",
                        names=['t'])['t'].tolist()

#中文分词函数
def txt_cut(juzi):
    return [w for w in jieba.lcut(juzi) if w not in stop_list]

#写入分词结果
fw = open('fenci_data.csv', "a+", newline = '',encoding = 'gb18030')
writer = csv.writer(fw)  
writer.writerow(['content','label'])

# 使用csv.DictReader读取文件中的信息
labels = []
contents = []
file = "data.csv"
with open(file, "r", encoding="UTF-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 数据元素获取
        if row['label'] == '好评':
            res = 0
        else:
            res = 1
        labels.append(res)
        content = row['content']
        seglist = txt_cut(content)
        output = ' '.join(list(seglist))            #空格拼接
        contents.append(output)
        
        #文件写入
        tlist = []
        tlist.append(output)
        tlist.append(res)
        writer.writerow(tlist)

print(labels[:5])
print(contents[:5])
fw.close()
