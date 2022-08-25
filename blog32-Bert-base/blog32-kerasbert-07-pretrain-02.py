# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 00:09:48 2021
@author: xiuzhang
"""
import os
from keras_bert import load_vocabulary
from keras_bert import Tokenizer
from keras_bert import load_trained_model_from_checkpoint
import numpy as np

#-------------------------------第一步 加载模型--------------------------------- 
#设置预训练模型的路径
pretrained_path = 'chinese_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')
 
#构建字典
token_dict = load_vocabulary(vocab_path)
print(token_dict)
print(len(token_dict))

#Tokenization
tokenizer = Tokenizer(token_dict)
print(tokenizer)

#加载预训练模型
model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
print(model)

#-------------------------------第二步 特征提取--------------------------------- 
text = '语言模型'
tokens = tokenizer.tokenize(text)
print(tokens)
#['[CLS]', '语', '言', '模', '型', '[SEP]']

indices, segments = tokenizer.encode(first=text, max_len=512)
print(indices[:10])
print(segments[:10])
 
#提取特征
predicts = model.predict([np.array([indices]), np.array([segments])])[0]
for i, token in enumerate(tokens):
    print(token, predicts[i].tolist()[:5])
