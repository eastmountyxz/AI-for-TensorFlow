#encoding:utf-8
# By: Eastmount WuShuai 2024-02-05
import re
import os
import csv
import sys

train_data_path = "data/train.csv"
test_data_path = "data/test.csv"
char_vocab_path = "char_vocabs.txt"    #字典文件
special_words = ['<PAD>', '<UNK>']     #特殊词表示
final_words = []                       #统计词典（不重复出现）
final_labels = []                      #统计标记（不重复出现）

#语料文件读取函数
def read_csv(filename):
    words = []
    labels = []
    with open(filename,encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row)>0: #存在空行报错越界
                word,label = row[0],row[1]
                words.append(word)
                labels.append(label)
    return words,labels

#统计不重复词典
def count_vocab(words,labels):
    fp = open(char_vocab_path, 'a') #注意a为叠加（文件只能运行一次）
    k = 0
    while k<len(words):
        word = words[k]
        label = labels[k]
        if word not in final_words:
            final_words.append(word)
            fp.writelines(word + "\n")
        if label not in final_labels:
            final_labels.append(label)
        k += 1
    fp.close()
   
#读取数据并构造原文字典（第一列）
def build_vocab():
    words,labels = read_csv(train_data_path)
    print(len(words),len(labels),words[:8],labels[:8])
    count_vocab(words,labels)
    print(len(final_words),len(final_labels))

    #测试集
    words,labels = read_csv(test_data_path)
    print(len(words),len(labels))
    count_vocab(words,labels)
    print(len(final_words),len(final_labels))
    print(final_labels)

    #labels生成字典
    label_dict = {}
    k = 0
    for value in final_labels:
        label_dict[value] = k
        k += 1
    print(label_dict)
    return label_dict
    
if __name__ == '__main__':
    build_vocab()
     
