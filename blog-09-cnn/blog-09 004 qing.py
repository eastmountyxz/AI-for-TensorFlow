# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:48:50 2019
@author: xiuzhang Eastmount CSDN
"""
import os
import jieba
from gensim.models import Word2Vec
from gensim.models import word2vec

#----------------------------------中文分词----------------------------------
# 定义中文分词后文件名
file_name = "庆余年.txt"
cut_file  = "庆余年_cut.txt"
# 文件读取操作
f = open(file_name, 'r', encoding='utf-8')
text = f.read()
# Jieba分词
new_text = jieba.cut(text, cut_all=False)     #精确模式
# 过滤标点符号
str_out = ' '.join(new_text).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        .replace('’', '')    
# 输出文件
fo = open(cut_file, 'w', encoding='utf-8')
# 写入操作
fo.write(str_out)
f.close()
fo.close()


#----------------------------------训练模型----------------------------------  
save_model_name = '庆余年.model'
# 判断训练的模型文件是否存在
if not os.path.exists(save_model_name):            # 模型训练 
    sentences = word2vec.Text8Corpus(cut_file)     # 加载语料
    model = Word2Vec(sentences, size=200)          # 训练skip-gram模型
    model.save(save_model_name)
    # 二进制类型保存模型 后续直接使用
    model.wv.save_word2vec_format(save_model_name + ".bin", binary=True) 
else:
    print('此训练模型已经存在，不用再次训练')
   
#----------------------------------预测结果----------------------------------  
# 加载已训练好的模型
model = Word2Vec.load(save_model_name)

# 计算两个词的相似度/相关程度
res1 = model.wv.similarity("范闲", "林婉儿")
print(u"范闲和林婉儿的相似度为：", res1, "\n")

# 计算某个词的相关词列表
res2 = model.wv.most_similar("范闲", topn=10)  # 10个最相关的
print(u"和 [范闲] 最相关的词有：\n")
for item in res2:
    print(item[0], item[1])
print("-------------------------------\n")

# 计算某个词的相关词列表
res3 = model.wv.most_similar("五竹", topn=10)  # 10个最相关的
print(u"和 [五竹] 最相关的词有：\n")
for item in res3:
    print(item[0], item[1])
print("-------------------------------\n")
