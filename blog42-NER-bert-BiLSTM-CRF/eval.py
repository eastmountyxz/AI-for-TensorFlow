#encoding:utf-8
# By: Eastmount 2024-02-15
# 参考：https://www.bilibili.com/video/BV1KZ4y1z7Bx （每天都要机器学习）
# 版本：python 3.7, tf 2.2.0,  keras 2.3.1, bert4keras 0.11.5
import numpy as np
from ner_train import *
from data_utils import load_data

#预测实体标签
def predict_label(data):
    y_pred = []
    for d in data:
        text = ''.join([i[0] for i in d])
        pred = NER.recognize(text)

        #标签初始化为O 后续BIO替换
        label = ['O' for _ in range(len(text))]
        b = 0
        for item in pred:
            word,typ = item[0],item[1]
            start = text.find(word,b)
            end = start + len(word)
            label[start] = 'B-' + typ
            for i in range(start+1, end):
                label[i] = 'I-' + typ
            b += len(word)
        y_pred.append(label)
    return y_pred

#评估结果
def evaluate():
    test_data_path = "data/test_2w.csv"
    max_len = 70
    test_data,y_true = load_data(test_data_path,max_len)
    y_pred = predict_label(test_data)
    print(len(test_data),len(y_true),len(y_pred))

    #计算四个评价指标
    k = 0
    while k<len(y_true):
        print(len(test_data[k]), test_data[k])
        print(len(y_true[k]), y_true[k])
        print(len(y_pred[k]), y_pred[k])
        k += 1
        break
    
if __name__ == '__main__':
    evaluate()
