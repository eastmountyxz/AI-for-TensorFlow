#! -*- coding: utf-8 -*-
# By: Eastmount 2024-03-20
# 版本：python 3.7, tf 2.2.0,  keras 2.3.1, bert4keras 0.11.5
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm

#------------------------------------------------------------------------
# 参数设置
#------------------------------------------------------------------------
maxlen = 256
epochs = 4
batch_size = 32
bert_layers = 12
learning_rate = 2e-5
crf_lr_multiplier = 1000
categories = set()

# bert配置
config_path = './chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'

#------------------------------------------------------------------------
# 加载数据
# 单条格式：[text, (start, end, label), (start, end, label), ...]
# text[start:end+1]是类型为label的实体
#------------------------------------------------------------------------
def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split(' ')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                    categories.add(flag[2:])
                elif flag[0] == 'I':
                    d[-1][1] = i
                    D.append(d)
    return D

# 标注数据
train_data = load_data('./china-people-daily-ner-corpus/example-test.train')
valid_data = load_data('./china-people-daily-ner-corpus/example-test.dev')
test_data = load_data('./china-people-daily-ner-corpus/example-test.test')
categories = list(sorted(categories))
print(categories) #['LOC', 'ORG', 'PER']

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

#------------------------------------------------------------------------
# 构建数据生成器
#------------------------------------------------------------------------
class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens) 
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros(len(token_ids))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[start] = categories.index(label) * 2 + 1
                    labels[start + 1:end + 1] = categories.index(label) * 2 + 2
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


#------------------------------------------------------------------------
# 构建Bert-CRF模型
#------------------------------------------------------------------------
model = build_transformer_model(
    config_path,
    checkpoint_path,
)
output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = model.get_layer(output_layer).output
output = Dense(len(categories) * 2 + 1)(output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)
model = Model(model.input, output)
model.summary()
model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learning_rate),
    metrics=[CRF.sparse_accuracy]
)

#------------------------------------------------------------------------
# 构建命名实体识别器
#------------------------------------------------------------------------
class NamedEntityRecognizer(ViterbiDecoder):
    def recognize(self, text):
        tokens = tokenizer.tokenize(text, maxlen=512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], categories[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(mapping[w[0]][0], mapping[w[-1]][-1], l) for w, l in entities]

NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])

#------------------------------------------------------------------------
# 评测函数
#------------------------------------------------------------------------
def evaluate(data):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    n = 0
    for d in data:
        R = set(NER.recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        if n<=10:
            n += 1
            print("原文:",d)      #[['秦伯', 'PER'], ['使醫', 'O'], ['緩', 'PER'], ['爲之。', 'O']]
            print("R:",R)         #('緩', 'LOC'), ('伯', 'LOC'), ('。', 'LOC')}
            print("T:",T,"\n")    #{('緩','PER'),('秦伯','PER')}

        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

# 评估与保存
class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0
    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        print(NER.trans)
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./best_model.weights')
        print('valid: f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' %
              (f1, precision, recall, self.best_val_f1))
        f1, precision, recall = evaluate(test_data)
        print('test: f1: %.4f, precision: %.4f, recall: %.4f\n' %
              (f1, precision, recall))

#------------------------------------------------------------------------
# 主函数
#------------------------------------------------------------------------
if __name__ == '__main__':
    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
else:
    model.load_weights('./best_model.weights')
    NER.trans = K.eval(CRF.trans)
