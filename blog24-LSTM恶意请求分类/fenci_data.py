# coding: utf-8
import pandas as pd
import jieba
import time
import csv
from sklearn.feature_extraction.text import TfidfVectorizer

#----------------------------------自定义函数 N-Gram处理--------------------------------
# tokenizer function, this will make 3 grams of each query
# www.foo.com/1 转换为 ['www','ww.','w.f','.fo','foo','oo.','o.c','.co','com','om/','m/1']
def get_ngrams(query):
    tempQuery = str(query)
    ngrams = []
    for i in range(0, len(tempQuery)-3+1):
        ngrams.append(tempQuery[i:i+3])
    return ngrams

#----------------------------------主函数 读取文件及预处理-------------------------------
if __name__ == '__main__':
    # 使用csv.DictReader读取文件中的信息
    file = "all_data_url_random.csv"
    with open(file, "r", encoding="UTF-8") as f:
        reader = csv.DictReader(f)
        labels = []
        contents = []
        for row in reader:
            # 数据元素获取
            labels.append(row['label'])
            contents.append(row['content'])
    print(labels[:10])
    print(contents[:10])

    #文件写入
    #数据划分 前10000-训练集 中间5000-测试集 后5000-验证集
    ctrain = open("all_data_url_random_fenci_train.csv", "a+", newline='', encoding='gb18030')
    writer1 = csv.writer(ctrain)
    writer1.writerow(["label","fenci"])
    
    ctest = open("all_data_url_random_fenci_test.csv", "a+", newline='', encoding='gb18030')
    writer2 = csv.writer(ctest)
    writer2.writerow(["label","fenci"])
    
    cval = open("all_data_url_random_fenci_val.csv", "a+", newline='', encoding='gb18030')
    writer3 = csv.writer(cval)
    writer3.writerow(["label","fenci"])
    
    n = 0
    while n < len(contents):
        res = get_ngrams(contents[n])
        #print(res)
        final = ' '.join(res)
        tlist = []
        tlist.append(labels[n])
        tlist.append(final)
        if n<10000:
            writer1.writerow(tlist)  #训练集
        elif n>=10000 and n<15000:
            writer2.writerow(tlist)  #测试集
        elif n>=15000:
            writer3.writerow(tlist)  #验证集
        n = n + 1
    
    #文件关闭
    ctrain.close()
    ctest.close()
    cval.close()

