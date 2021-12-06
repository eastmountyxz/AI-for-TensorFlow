# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 00:09:02 2021
@author: xiuzhang
引用：https://github.com/percent4/keras_bert_text_classification
"""
import json
import numpy as np
import pandas as pd
from keras.models import load_model
from keras_bert import get_custom_objects
from sklearn.metrics import classification_report
from blog33_kerasbert_01_train import token_dict, OurTokenizer

maxlen = 300

#加载训练好的模型
model = load_model("cls_cnews.h5", custom_objects=get_custom_objects())
tokenizer = OurTokenizer(token_dict)
with open("label.json", "r", encoding="utf-8") as f:
    label_dict = json.loads(f.read())

#单句预测
def predict_single_text(text):
    text = text[:maxlen]
    x1, x2 = tokenizer.encode(first=text)  #BERT Tokenize
    X1 = x1 + [0] * (maxlen - len(x1)) if len(x1) < maxlen else x1
    X2 = x2 + [0] * (maxlen - len(x2)) if len(x2) < maxlen else x2
    #print(X1,X2)

    #模型预测
    predicted = model.predict([[X1], [X2]])
    y = np.argmax(predicted[0])
    return label_dict[str(y)]

#模型评估
def evaluate():
    test_df = pd.read_csv("data/sougou_mini/test.csv").fillna(value="")
    true_y_list, pred_y_list = [], []
    for i in range(test_df.shape[0]):
        print("predict %d samples" % (i+1))
        true_y, content = test_df.iloc[i, :]
        pred_y = predict_single_text(content)
        print(true_y,pred_y)
        true_y_list.append(true_y)
        pred_y_list.append(pred_y)
    return classification_report(true_y_list, pred_y_list, digits=4)

#------------------------------------模型评估---------------------------------
output_data = evaluate()
print("model evaluate result:\n")
print(output_data)
