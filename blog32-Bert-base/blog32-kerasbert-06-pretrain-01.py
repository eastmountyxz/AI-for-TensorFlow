# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 00:09:48 2021
@author: xiuzhang
"""
import os
from keras_bert import load_vocabulary
from keras_bert import Tokenizer
from keras_bert import load_trained_model_from_checkpoint
 
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
