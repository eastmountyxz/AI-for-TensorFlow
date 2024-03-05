#encoding:utf-8
# By: Eastmount 2024-02-15
# 参考：https://www.bilibili.com/video/BV1KZ4y1z7Bx （每天都要机器学习）
# 版本：python 3.7, tf 2.2.0,  keras 2.3.1, bert4keras 0.11.5
import os
import sys
import random
import pickle
import tensorflow as tf
from bert4keras.backend import K,keras,search_layer
from bert4keras.snippets import ViterbiDecoder, to_array

#加载定义好的函数
from data_utils import *
from build_model import bert_bilstm_crf

seed = 233
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

epochs = 1
max_len = 70
batch_size = 64
lstm_units = 64
drop_rate = 0.1
learning_rate = 0.001
config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
checkpoint_save_path = 'checkpoint/bert_bilstm_crf.weights'

#------------------------------------------------------
#命名实体识别器
#------------------------------------------------------
class NamedEntityRecognizer(ViterbiDecoder):
    #识别句子中的实体类别
    def recognize(self,text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > max_len:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids,segment_ids = to_array([token_ids],[segment_ids]) #ndarray
        #CRF识别后得到概率转移矩阵
        nodes = model.predict([token_ids,segment_ids])[0] #[seq_len,7]
        #利用维特比算法解码概率转移矩阵 标签一维列表
        labels = self.decode(nodes) #id [seq_len,] [0 0 0 2 3 0 0]
        entities,starting = [], False
        #标签ID => 解析成标签
        for i,label in enumerate(labels):
            if label>0:
                if label % 2 ==1:
                    starting = True
                    entities.append([[i], id2label[(label-1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        #判断识别实体的具体位置
        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1],k) for w,k in entities]
        
#实例化
model,crf = bert_bilstm_crf(config_path,checkpoint_path,num_labels,
                            lstm_units,drop_rate,learning_rate)
#初始化参数为概率转移矩阵 其它默认0
NER = NamedEntityRecognizer(trans=K.eval(crf.trans), starts=[0], ends=[0])

if __name__ == '__main__':
    train_data_path = "data/train_2w.csv"
    val_data_path = "data/val_2w.csv"
    char_vocab_path = "char_vocabs_.txt"

    train_data,train_label = load_data(train_data_path,max_len)
    print(len(train_data),len(train_label))
    val_data,val_label = load_data(val_data_path,max_len)
    print(len(val_data),len(val_label))

    #数据迭代器
    train_generator = data_generator(train_data,batch_size)
    valid_generator = data_generator(val_data,batch_size*5)

    #检测到验证集准确率最大时保存模型
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_save_path,
        monitor = 'val_sparse_accuracy',
        verbose = 1,
        save_best_only = True,
        mode = 'max'
        )
    
    #模型训练
    model.fit(
        train_generator.forfit(),
        steps_per_epoch = len(train_generator),
        validation_data = valid_generator.forfit(),
        validation_steps = len(valid_generator),
        epochs = epochs,
        callbacks = [checkpoint]
        )
    #model.save("bert_bilstm_ner_model.h5")
    #保存CRF概率转移矩阵
    print(K.eval(crf.trans))
    print(K.eval(crf.trans).shape)
    pickle.dump(K.eval(crf.trans),open('checkpoint/crf_trans.pkl','wb'))
else:
    model.load_weights(checkpoint_save_path)
    NER.trans = pickle.load(open('checkpoint/crf_trans.pkl','rb'))
    
