# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 21:00:40 2021
@author: xiuzhang
"""
from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths

#下载解压数据
model_path = get_pretrained(PretrainedList.multi_cased_base)
paths = get_checkpoint_paths(model_path)
print(paths.config, paths.checkpoint, paths.vocab)
