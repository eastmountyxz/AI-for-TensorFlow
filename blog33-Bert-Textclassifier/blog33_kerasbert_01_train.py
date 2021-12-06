# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 00:09:48 2021
@author: xiuzhang
"""
import json
import codecs
import pandas as pd
import numpy as np
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

import os
import tensorflow as tf
os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#指定了每个GPU进程中使用显存的上限,0.9表示可以使用GPU 90%的资源进行训练
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

maxlen = 300
BATCH_SIZE = 8
config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

#读取vocab词典
token_dict = {}
with codecs.open(dict_path, 'r', 'utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

#------------------------------------------类函数定义--------------------------------------
#词典中添加否则Unknown
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            else:
                R.append('[UNK]')   #剩余的字符是[UNK]
        return R
tokenizer = OurTokenizer(token_dict)

#数据填充
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

class DataGenerator:

    def __init__(self, data, batch_size=BATCH_SIZE):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []

#构建模型
def create_cls_model(num_labels):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for layer in bert_model.layers:
        layer.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    cls_layer = Lambda(lambda x: x[:, 0])(x)                #取出[CLS]对应的向量用来做分类
    p = Dense(num_labels, activation='softmax')(cls_layer)  #多分类

    model = Model([x1_in, x2_in], p)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),
        metrics=['accuracy']
    )
    model.summary()

    return model

#------------------------------------------主函数-----------------------------------------
if __name__ == '__main__':

    #数据预处理
    train_df = pd.read_csv("data/sougou_mini/train.csv").fillna(value="")
    test_df = pd.read_csv("data/sougou_mini/test.csv").fillna(value="")
    print("begin data processing...")

    labels = train_df["label"].unique()
    with open("label.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(zip(range(len(labels)), labels)), ensure_ascii=False, indent=2))

    train_data = []
    test_data = []
    for i in range(train_df.shape[0]):
        label, content = train_df.iloc[i, :]
        label_id = [0] * len(labels)
        for j, _ in enumerate(labels):
            if _ == label:
                label_id[j] = 1
        train_data.append((content, label_id))
    print(train_data[0])

    for i in range(test_df.shape[0]):
        label, content = test_df.iloc[i, :]
        label_id = [0] * len(labels)
        for j, _ in enumerate(labels):
            if _ == label:
                label_id[j] = 1
        test_data.append((content, label_id))
    print(len(train_data),len(test_data))
    print("finish data processing!\n")
    
    #模型训练
    model = create_cls_model(len(labels))
    train_D = DataGenerator(train_data)
    test_D = DataGenerator(test_data)
    print("begin model training...")
    
    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=3,
        validation_data=test_D.__iter__(),
        validation_steps=len(test_D)
    )
    print("finish model training!")

    #模型保存
    model.save('cls_cnews.h5')
    print("Model saved!")
    result = model.evaluate_generator(test_D.__iter__(), steps=len(test_D))
    print("模型评估结果:", result)

