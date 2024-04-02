#encoding:utf-8
import re
import os
import csv

#------------------------------------------------------------------------
#第一步 生成词典
#------------------------------------------------------------------------
train_data_path = "dataset-train.txt"  #训练数据
test_data_path = "dataset-test.txt"    #测试数据
val_data_path = "dataset-val.txt"      #验证数据

char_vocab_path = "char_vocabs.txt"    #字典文件
words = []
contents = ""


#训练数据处理
f = open(train_data_path, "r", encoding="utf8")
fw = open(char_vocab_path, "w", encoding="utf8")

for con in f.readlines():
    con = con.strip()
    text = con.split()
    if len(text)>1:
        word = text[0]
        label = text[1]
        #print(word,label)
        if word!="" and word not in words:
            words.append(word)
            contents += word + "\n"
print(contents)
fw.write(contents)
f.close()


#测试数据处理
contents = ""
f = open(test_data_path, "r", encoding="utf8")
for con in f.readlines():
    con = con.strip()
    text = con.split()
    if len(text)>1:
        word = text[0]
        label = text[1]
        #print(word,label)
        if word!="" and word not in words:
            words.append(word)
            contents += word + "\n"
print(contents)
fw.write(contents)
f.close()


#验证数据处理
contents = ""
f = open(val_data_path, "r", encoding="utf8")
for con in f.readlines():
    con = con.strip()
    text = con.split()
    if len(text)>1:
        word = text[0]
        label = text[1]
        #print(word,label)
        if word!="" and word not in words:
            words.append(word)
            contents += word + "\n"
print(contents)
fw.write(contents)
f.close()
fw.close()

