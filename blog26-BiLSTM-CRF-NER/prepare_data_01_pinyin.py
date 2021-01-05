#encoding:utf-8
import os
import pandas as pd
from collections import Counter
from data_process import split_text
from tqdm import tqdm          #进度条 pip install tqdm 
#词性标注
import jieba.posseg as psg
#获取字的偏旁和拼音
from cnradical import Radical, RunOption


train_dir = "train_data"

#----------------------------功能:文本预处理---------------------------------
train_dir = "train_data"

def process_text(idx, split_method=None):
    """
    功能: 读取文本并切割,接着打上标记及提取词边界、词性、偏旁部首、拼音等特征
    param idx: 文件的名字 不含扩展名
    param split_method: 切割文本方法
    return
    """

    #定义字典 保存所有字的标记、边界、词性、偏旁部首、拼音等特征
    data = {}

    #--------------------------------------------------------------------
    #获取句子
    #--------------------------------------------------------------------
    if split_method is None:
        #未给文本分割函数 -> 读取文件
        with open(f'data/{train_dir}/{idx}.txt', encoding='utf8') as f:     #f表示文件路径
            texts = f.readlines()
    else:
        #给出文本分割函数 -> 按函数分割
        with open(f'data/{train_dir}/{idx}.txt', encoding='utf8') as f:
            outfile = f'data/train_data_pro/{idx}_pro.txt'
            print(outfile)
            texts = f.read()
            texts = split_method(texts, outfile)

    #提取句子
    data['word'] = texts
    print(texts)

    #--------------------------------------------------------------------
    #                             获取标签
    #--------------------------------------------------------------------
    #初始时将所有汉字标记为O
    tag_list = ['O' for s in texts for x in s]    #双层循环遍历每句话中的汉字

    #读取ANN文件获取每个实体的类型、起始位置和结束位置
    tag = pd.read_csv(f'data/{train_dir}/{idx}.ann', header=None, sep='\t') #Pandas读取 分隔符为tab键
    #0 T1 Disease 1845 1850  1型糖尿病

    for i in range(tag.shape[0]):  #tag.shape[0]为行数
        tag_item = tag.iloc[i][1].split(' ')    #每一行的第二列 空格分割
        #print(tag_item)
        #存在某些实体包括两段位置区间 仅获取起始位置和结束位置
        cls, start, end = tag_item[0], int(tag_item[1]), int(tag_item[-1])
        #print(cls,start,end)
        
        #对tag_list进行修改
        tag_list[start] = 'B-' + cls
        for j in range(start+1, end):
            tag_list[j] = 'I-' + cls

    #断言 两个长度不一致报错
    assert len([x for s in texts for x in s])==len(tag_list)
    #print(len([x for s in texts for x in s]))
    #print(len(tag_list))

    #--------------------------------------------------------------------
    #                       分割后句子匹配标签
    #--------------------------------------------------------------------
    tags = []
    start = 0
    end = 0
    #遍历文本
    for s in texts:
        length = len(s)
        end += length
        tags.append(tag_list[start:end])
        start += length    
    print(len(tags))
    #标签数据存储至字典中
    data['label'] = tags

    #--------------------------------------------------------------------
    #                       提取词性和词边界
    #--------------------------------------------------------------------
    #初始标记为M
    word_bounds = ['M' for item in tag_list]    #边界 M表示中间
    word_flags = []                             #词性
    
    #分词
    for text in texts:
        #带词性的结巴分词
        for word, flag in psg.cut(text):   
            if len(word)==1:  #1个长度词
                start = len(word_flags)
                word_bounds[start] = 'S'   #单个字
                word_flags.append(flag)
            else:
                start = len(word_flags)
                word_bounds[start] = 'B'         #开始边界
                word_flags += [flag]*len(word)   #保证词性和字一一对应
                end = len(word_flags) - 1
                word_bounds[end] = 'E'           #结束边界
    #存储
    bounds = []
    flags = []
    start = 0
    end = 0
    for s in texts:
        length = len(s)
        end += length
        bounds.append(word_bounds[start:end])
        flags.append(word_flags[start:end])
        start += length
    data['bound'] = bounds
    data['flag'] = flags

    #--------------------------------------------------------------------
    #                             获取拼音特征
    #--------------------------------------------------------------------
    radical = Radical(RunOption.Radical)   #提取偏旁部首
    pinyin = Radical(RunOption.Pinyin)     #提取拼音

    #提取拼音和偏旁 None用特殊符号替代
    radical_out = [[radical.trans_ch(x) if radical.trans_ch(x) is not None else 'PAD' for x in s] for s in texts]
    pinyin_out = [[pinyin.trans_ch(x) if pinyin.trans_ch(x) is not None else 'PAD' for x in s] for s in texts]

    #赋值
    data['radical'] = radical_out
    data['pinyin'] = pinyin_out
    
    #return texts, tags, bounds, flags
    return texts[0], tags[0], bounds[0], flags[0], radical_out[0], pinyin_out[0]

#-------------------------------功能:主函数--------------------------------------
if __name__ == '__main__':
    print(process_text('0',split_method=split_text))



