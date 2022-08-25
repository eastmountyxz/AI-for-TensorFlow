# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 00:10:06 2021
@author: xiuzhang
引用：https://github.com/percent4/keras_bert_text_classification
"""
import time
import json
import numpy as np

from blog34_kerasbert_train import token_dict, OurTokenizer
from keras.models import load_model
from keras_bert import get_custom_objects

maxlen = 256
s_time = time.time()

#加载训练好的模型
model = load_model("cls_mood.h5", custom_objects=get_custom_objects())
tokenizer = OurTokenizer(token_dict)
with open("label.json", "r", encoding="utf-8") as f:
    label_dict = json.loads(f.read())

#预测示例语句
text = "《长津湖》这部电影真的非常好看，今天看完好开心，爱了爱了。强烈推荐大家，哈哈！！！"
#text = "听到这个消息真心难受，很伤心，怎么这么悲剧。保佑保佑，哭"
#text = "愤怒，我真的挺生气的，怒其不争，哀其不幸啊！"

#Tokenize
text = text[:maxlen]
x1, x2 = tokenizer.encode(first=text)
X1 = x1 + [0] * (maxlen-len(x1)) if len(x1) < maxlen else x1
X2 = x2 + [0] * (maxlen-len(x2)) if len(x2) < maxlen else x2

#模型预测
predicted = model.predict([[X1], [X2]])
y = np.argmax(predicted[0])
e_time = time.time()
print("原文: %s" % text)
print("预测标签: %s" % label_dict[str(y)])
print("Cost time:", e_time-s_time)
