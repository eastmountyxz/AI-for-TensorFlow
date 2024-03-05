#encoding:utf-8
# By: Eastmount 2024-02-15
# 参考：https://www.bilibili.com/video/BV1KZ4y1z7Bx （每天都要机器学习）
# 版本：python 3.7, tf 2.2.0,  keras 2.3.1, bert4keras 0.11.5
import re
import csv
import numpy as np
import pandas as pd
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding,DataGenerator

#------------------------------------------------------------------------
#第一步 数据预处理
#------------------------------------------------------------------------
train_data_path = "data/train_2w.csv"
char_vocab_path = "char_vocabs_.txt"   #字典文件
vocab_path = "chinese_L-12_H-768_A-12/vocab.txt"

"""
#BIO标记的标签 字母O初始标记为0
label2idx = {'O': 0,
             'S-LOC': 1, 'B-LOC': 2,  'I-LOC': 3,  'E-LOC': 4,
             'S-PER': 5, 'B-PER': 6,  'I-PER': 7,  'E-PER': 8,
             'S-TIM': 9, 'B-TIM': 10, 'E-TIM': 11, 'I-TIM': 12
             }
#{'S-LOC': 0, 'B-PER': 1, 'I-PER': 2, ...., 'I-TIM': 11, 'I-LOC': 12}

#索引和BIO标签对应
idx2label = {idx: label for label, idx in label2idx.items()}
#{0: 'S-LOC', 1: 'B-PER', 2: 'I-PER', ...., 11: 'I-TIM', 12: 'I-LOC'}
"""

#实体标签
entity_labels = ['LOC','PER','TIM']
id2label = {i:j for i,j in enumerate(sorted(entity_labels))}
print(id2label) #{0: 'LOC', 1: 'PER', 2: 'TIM'}
label2id = {j:i for i,j in id2label.items()}
num_labels = len(entity_labels) * 2 + 1 #BIO

#词典Tokenizer
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
print(tokenizer)

#------------------------------------------------------------------------
#第二步 数据读取
#------------------------------------------------------------------------
def load_data(data_path,max_len):
    #数据格式:[(片段1,标签1),(片段2,标签2),...]
    datasets = []
    samples_len = []
    X = []                 #单词
    y = []                 #类别
    sentence = []
    label = []
    #分隔符断句 设计一个最大长度
    split_pattern = re.compile(r'[;；。，、？！\.\n\r\?,! ]')
    with open(data_path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            #每行一个字符和tag
            #sentence=[w1,w2,...,wn] labels=[B-xx,I-xx,...,O]
            line = line.strip().split(',')
            if (not line or len(line)<2):
                X.append(sentence.copy())
                y.append(label.copy())
                sentence.clear()
                label.clear()
                continue
            word, tag = line[0], line[1]
            if split_pattern.match(word) and len(sentence) >= max_len:
                sentence.append(word)
                label.append(tag)
                X.append(sentence.copy())
                y.append(label.copy())
                sentence.clear()
                label.clear()
            else:
                sentence.append(word)
                label.append(tag)
    if len(sentence):
        X.append(sentence.copy())
        y.append(label.copy())
        sentence.clear()
        label.clear()

    #转换成序列 形如sample_seq=[['xxx','B-xx'],['yyy','I-yy'],[],...]
    for token_seq,label_seq in zip(X,y):
        if len(token_seq) < 2:
            continue
        sample_seq,last_flag = [],''
        for token, flag in zip(token_seq,label_seq):
            #换句处理
            if token=="" and flag=="":
                continue
            #BMES标注 => BIO标注
            fchar = flag[0]
            fchar = fchar.replace('M','I').replace('E','I').replace('S','B')
            flag = fchar + flag[1:]
            if flag=='O' and last_flag=='O':
                sample_seq[-1][0] += token
            elif flag=='O' and last_flag!='O':
                sample_seq.append([token,'O'])
            elif flag[:1]=='B':
                sample_seq.append([token, flag[2:]]) #B-LOC 标签只取LOC
            else:
                if sample_seq:
                    sample_seq[-1][0] += token
            last_flag = flag

        datasets.append(sample_seq)
        samples_len.append(len(token_seq))
                
    df = pd.DataFrame(samples_len)
    print(df.describe())
    return datasets,y

#------------------------------------------------------------------------
#第三步 数据生成器 继承DataGenerator
#tokenizer将字符转换为vocab.txt中的索引
#------------------------------------------------------------------------
class data_generator(DataGenerator):
    def __iter__(self,random=True):
        batch_token_ids, batch_segment_ids, batch_labels = [],[],[]
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0] #[CLS]
            for w,l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < 70:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id] #[seq]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            #训练完执行 yield是生成器（generator） 返回Bert两个输入和标签
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [],[],[]
    
if __name__ == '__main__':
    data,y = load_data(train_data_path,128)
    print(data[:3])

    #统计三类实体Top10 合并数据集
    fw = open('result_entity.csv','w',newline='',encoding='utf-8')
    writer = csv.writer(fw)
    for data_list in data:
        for value in data_list:
            #print(value[0],value[1])
            writer.writerow([value[0],value[1]])
    fw.close()
    
