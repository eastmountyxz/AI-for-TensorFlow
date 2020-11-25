# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:48:50 2019
@author: xiuzhang Eastmount CSDN

博客原地址：https://blog.csdn.net/Yellow_python/article/details/86726619
推荐大家学习Yellow_python大神的文章
"""
from gensim.models import Word2Vec  # 词向量
from random import choice
from os.path import exists

class CONF:
    path = '古诗词.txt'
    window = 16           # 滑窗大小
    min_count = 60        # 过滤低频字
    size = 125            # 词向量维度
    topn = 14             # 生成诗词的开放度
    model_path = 'word2vec'

# 定义模型
class Model:
    def __init__(self, window, topn, model):
        self.window = window
        self.topn = topn
        self.model = model  # 词向量模型
        self.chr_dict = model.wv.index2word  # 字典

    """模型初始化"""
    @classmethod
    def initialize(cls, config):
        if exists(config.model_path):
            # 模型读取
            model = Word2Vec.load(config.model_path)
        else:
            # 语料读取
            with open(config.path, encoding='utf-8') as f:
                ls_of_ls_of_c = [list(line.strip()) for line in f]
            # 模型训练和保存
            model = Word2Vec(sentences=ls_of_ls_of_c, size=config.size, window=config.window, min_count=config.min_count)
            model.save(config.model_path)
        return cls(config.window, config.topn, model)

    """古诗词生成"""
    def poem_generator(self, title, form):
        filter = lambda lst: [t[0] for t in lst if t[0] not in ['，', '。']]
        # 标题补全
        if len(title) < 4:
            if not title:
                title += choice(self.chr_dict)
            for _ in range(4 - len(title)):
                similar_chr = self.model.similar_by_word(title[-1], self.topn // 2)
                similar_chr = filter(similar_chr)
                char = choice([c for c in similar_chr if c not in title])
                title += char
                
        # 文本生成
        poem = list(title)
        for i in range(form[0]):
            for _ in range(form[1]):
                predict_chr = self.model.predict_output_word(poem[-self.window:], max(self.topn, len(poem) + 1))
                predict_chr = filter(predict_chr)
                char = choice([c for c in predict_chr if c not in poem[len(title):]])
                poem.append(char)
            poem.append('，' if i % 2 == 0 else '。')
        length = form[0] * (form[1] + 1)
        return '《%s》' % ''.join(poem[:-length]) + '\n' + ''.join(poem[-length:])


def main(config=CONF):
    form = {'五言绝句': (4, 5), '七言绝句': (4, 7), '对联': (2, 9)}
    m = Model.initialize(config)
    while True:
        title = input('输入标题：').strip()
        try:
            poem = m.poem_generator(title, form['五言绝句'])
            print('%s' % poem)  # red
            poem = m.poem_generator(title, form['七言绝句'])
            print('%s' % poem)  # yellow
            poem = m.poem_generator(title, form['对联'])
            print('%s' % poem)  # purple
            print()
        except:
            pass

if __name__ == '__main__':
    main()