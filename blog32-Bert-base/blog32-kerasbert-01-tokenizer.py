# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:35:48 2021
@author: xiuzhang
"""
from keras_bert import Tokenizer

token_dict = {
    '[CLS]': 0,
    '[SEP]': 1,
    'un': 2,
    '##aff': 3,
    '##able': 4,
    '[UNK]': 5,
}

#分词器-Tokenizer
tokenizer = Tokenizer(token_dict)
print(tokenizer.tokenize('unaffable'))  

#拆分单词
indices, segments = tokenizer.encode('unaffable')
print(indices)    #字对应索引
print(segments)   #索引对应位置上字属于第一句话还是第二句话 
print(tokenizer.tokenize(first='unaffable', second='钢'))

indices, segments = tokenizer.encode(first='unaffable', second='钢', max_len=10)
print(indices)
print(segments)


