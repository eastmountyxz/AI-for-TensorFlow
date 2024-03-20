#encoding:utf-8
# By: Eastmount 2024-02-15
# 参考：https://www.bilibili.com/video/BV1KZ4y1z7Bx （每天都要机器学习）
#       https://github.com/dengxc1220/bert4keras_ner_demo/tree/master
# 版本：python 3.7, tf 2.2.0,  keras 2.3.1, bert4keras 0.11.5
import re, os, json
import numpy as np
from bert4keras.backend import keras, K
#from bert4keras.bert import build_bert_model
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Dropout
from keras.models import Model
from tqdm import tqdm

#------------------------------------------------------------------------
#第一步 数据预处理
#------------------------------------------------------------------------
#bert配置
config_path = './chinese_L-12_H-7 68_A-12/bert_config.json'
checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'

#实体标签
#classes = set(['PER', 'LOC', 'TIM'])
classes = set(['PER', 'LOC', 'ORG'])
id2class = dict(enumerate(classes))
class2id = {j: i for i, j in id2class.items()}
num_labels = len(classes) * 2 + 1 #BIO

#建立分词器 词典Tokenizer
tokenizer = Tokenizer(dict_path, do_lower_case=True)
print(tokenizer)

maxlen = 128
epochs = 1
batch_size = 128
bert_layers = 12
learing_rate = 1e-5
crf_lr_multiplier = 100


#------------------------------------------------------------------------
#第二步 数据读取
#       数据格式:[(片段1,标签1),(片段2,标签2),...]
#------------------------------------------------------------------------
def load_data(filename):
    datasets = []
    labels = []
    try:
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            for line in f.split('\n\n'): #分句
                if not line:
                    continue
                word, last_flag = [], ''
                label = []
                for c in line.split('\n'):
                    if not c: #c => 晉 S-LOC
                        continue
                    char, tag = c.split(' ')
                    #BMES标注 => BIO标注
                    this_flag = tag.replace('E-','I-').replace('M-','I-').replace('S-','B-')
                    label.append(this_flag)
                    if this_flag == 'O' and last_flag == 'O':
                        word[-1][0] += char
                    elif this_flag == 'O' and last_flag != 'O':
                        word.append([char, 'O'])
                    elif this_flag[:1] == 'B':
                        word.append([char, this_flag[2:]]) #B-LOC 标签只取LOC
                    else:
                        word[-1][0] += char
                    last_flag = this_flag
                #word => [['晉', 'LOC'], ['樂王鮒', 'PER'], ['曰：', 'O']]
                datasets.append(word)
                labels.append(label)
    except Exception as e:
        print(e)
    return datasets,labels

#------------------------------------------------------------------------
#第三步 数据生成器 继承DataGenerator
#       tokenizer将字符转换为vocab.txt中的索引
#------------------------------------------------------------------------
class data_generator(DataGenerator):
    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            token_ids, labels = [tokenizer._token_start_id], [0] #[CLS]
            for w, l in self.data[i]:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = class2id[l] * 2 + 1
                        I = class2id[l] * 2 + 2
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
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

#载入数据
"""
train_data,train_y = load_data('./data/train_2w.txt') #val测试
valid_data,valid_y = load_data('./data/val_2w.txt')
test_data,test_y = load_data('./data/test_2w.txt')
"""
train_data,train_y = load_data('./china-people-daily-ner-corpus/example-test.train')
valid_data,valid_y = load_data('./china-people-daily-ner-corpus/example-test.dev')
test_data,test_y = load_data('./china-people-daily-ner-corpus/example-test.test')
print(len(train_data),len(train_y))
print(len(valid_data),len(valid_y))
print(len(test_data),len(test_y))


#------------------------------------------------------------------------
#第四步 搭建模型
#       参数：Bert配置文件路径 LSTM单元数 dropout参数
#------------------------------------------------------------------------
#定义bert模型 使用bert自带输入
bert = build_transformer_model(
    config_path = config_path,
    checkpoint_path = checkpoint_path,
    model = 'bert',
    return_keras_model = False
)
#输出Bert编码的三维Embedding向量 每个Tokenizer都有一个768维词向量
x = bert.model.output #[batch_size, seq_length, 768]

#词向量传给BiLSTM模型 初始化及返回每个Tokenizer的输出
bilstm = Bidirectional(LSTM(64,return_sequences = True))(x)

#序列数据捕获时间信息
x = TimeDistributed(Dropout(0.3))(bilstm)

#全连接层 标签数量 [batch_size, seq_length, num_labels]
output = Dense(num_labels,activation='relu',kernel_initializer = 'he_normal')(x)

#构建CRF模型
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

#模型实例化及编译
model = Model(bert.input, output)
model.summary()
model.compile(loss=CRF.sparse_loss,
              optimizer=Adam(learing_rate),
              metrics=[CRF.sparse_accuracy])


#------------------------------------------------------------------------
#第五步 Viterbi算法求最优路径
#       nodes.shape=[seq_len, num_labels]
#       trans.shape=[num_labels, num_labels]
#------------------------------------------------------------------------
def viterbi_decode(nodes, trans):
    labels = np.arange(num_labels).reshape((1, -1))
    scores = nodes[0].reshape((-1, 1))
    scores[1:] -= np.inf  #第一个标签是0
    paths = labels
    for l in range(1, len(nodes)):
        M = scores + trans + nodes[l].reshape((1, -1))
        idxs = M.argmax(0)
        scores = M.max(0).reshape((-1, 1))
        paths = np.concatenate([paths[:, idxs], labels], 0)
    return paths[:, scores[0].argmax()]

#------------------------------------------------------------------------
#第六步 命名实体识别函数
#------------------------------------------------------------------------
def named_entity_recognize(text):
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 512:
        tokens.pop(-2)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    #CRF识别后得到概率转移矩阵
    nodes = model.predict([[token_ids], [segment_ids]])[0]
    trans = K.eval(CRF.trans)
    #利用维特比算法解码概率转移矩阵 标签一维列表
    labels = viterbi_decode(nodes, trans)[1:-1]
    #标签ID => 解析成标签
    entities, starting = [], False
    for token, label in zip(tokens[1:-1], labels):
        if label > 0:
            if label % 2 == 1:
                starting = True
                entities.append([[token], id2class[(label - 1) // 2]])
            elif starting:
                entities[-1][0].append(token)
            else:
                starting = False
        else:
            starting = False
    return [(tokenizer.decode(w, w).replace(' ', ''), l) for w, l in entities]


#------------------------------------------------------------------------
#第七步 评测函数
#------------------------------------------------------------------------
def evaluate(data):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    n = 0
    for d in data:
        text = ''.join([i[0] for i in d])
        label = [i[1] for i in d]
        result = named_entity_recognize(text)
        R = set(result)
        T = set([tuple(i) for i in d if i[1] != 'O'])
        
        if n<=10:
            n += 1
            print("原文:",d)      #[['秦伯', 'PER'], ['使醫', 'O'], ['緩', 'PER'], ['爲之。', 'O']]
            print("文本:",text)   #秦伯使醫緩爲之。
            print("标签:",label)  #['PER', 'O', 'PER', 'O']
            print("预测:",result) #[('伯', 'LOC'), ('緩', 'LOC'), ('。', 'LOC')]
            print("R:",R)         #('緩', 'LOC'), ('伯', 'LOC'), ('。', 'LOC')}
            print("T:",T,"\n")    #{('緩','PER'),('秦伯','PER')}
        
        #统计预测和真实结果类别相同
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        print(trans)
        f1, precision, recall = evaluate(valid_data)
        #保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./best_model.weights')
        print('valid:  f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' %
              (f1, precision, recall, self.best_val_f1))
        f1, precision, recall = evaluate(test_data)
        print('test:  f1: %.4f, precision: %.4f, recall: %.4f\n' %
              (f1, precision, recall))


#------------------------------------------------------------------------
#第八步 训练和测试
#------------------------------------------------------------------------
evaluator = Evaluate()
train_generator = data_generator(train_data, batch_size)
history = model.fit_generator(train_generator.forfit(),
                steps_per_epoch=len(train_generator),
                epochs=epochs,
                callbacks=[evaluator])

text = '秦伯使醫緩爲之。 六月丙午，晉侯欲麥，使甸人獻麥，饋人爲之。'
result = named_entity_recognize(text)
print(result)
