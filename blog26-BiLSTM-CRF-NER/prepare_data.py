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
#删除目录
import shutil
#随机划分训练集和测试集
from random import shuffle

train_dir = "train_data"

#----------------------------功能:文本预处理---------------------------------
train_dir = "train_data"

def process_text(idx, split_method=None, split_name='train'):
    """
    功能: 读取文本并切割,接着打上标记及提取词边界、词性、偏旁部首、拼音等特征
    param idx: 文件的名字 不含扩展名
    param split_method: 切割文本方法
    param split_name: 存储数据集 默认训练集, 还有测试集
    return
    """

    #定义字典 保存所有字的标记、边界、词性、偏旁部首、拼音等特征
    data = {}

    #--------------------------------------------------------------------
    #                            获取句子
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
    #                             获取标签(实体类别、起始位置)
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
    #                         获取拼音和偏旁特征
    #--------------------------------------------------------------------
    radical = Radical(RunOption.Radical)   #提取偏旁部首
    pinyin = Radical(RunOption.Pinyin)     #提取拼音

    #提取拼音和偏旁 None用特殊符号替代
    radical_out = [[radical.trans_ch(x) if radical.trans_ch(x) is not None else 'PAD' for x in s] for s in texts]
    pinyin_out = [[pinyin.trans_ch(x) if pinyin.trans_ch(x) is not None else 'PAD' for x in s] for s in texts]

    #赋值
    data['radical'] = radical_out
    data['pinyin'] = pinyin_out

    #--------------------------------------------------------------------
    #                              存储数据
    #--------------------------------------------------------------------
    #获取样本数量
    num_samples = len(texts)     #行数
    num_col = len(data.keys())   #列数 字典自定义类别数 6
    print(num_samples)
    print(num_col)
    
    dataset = []
    for i in range(num_samples):
        records = list(zip(*[list(v[i]) for v in data.values()]))   #压缩
        dataset += records+[['sep']*num_col]                        #每处理一句话sep分割
    #records = list(zip(*[list(v[0]) for v in data.values()]))
    #for r in records:
    #    print(r)
    
    #最后一行sep删除
    dataset = dataset[:-1]
    #转换成dataframe 增加表头
    dataset = pd.DataFrame(dataset,columns=data.keys())
    #保存文件 测试集 训练集
    save_path = f'data/prepare/{split_name}/{idx}.csv'
    dataset.to_csv(save_path,index=False,encoding='utf-8')

    #--------------------------------------------------------------------
    #                       处理换行符 w表示一个字
    #--------------------------------------------------------------------
    def clean_word(w):
        if w=='\n':
            return 'LB'
        if w in [' ','\t','\u2003']: #中文空格\u2003
            return 'SPACE'
        if w.isdigit():              #将所有数字转换为一种符号 数字训练会造成干扰
            return 'NUM'
        return w
    
    #对dataframe应用函数
    dataset['word'] = dataset['word'].apply(clean_word)

    #存储数据
    dataset.to_csv(save_path,index=False,encoding='utf-8')
    
    
    #return texts, tags, bounds, flags
    #return texts[0], tags[0], bounds[0], flags[0], radical_out[0], pinyin_out[0]


#----------------------------功能:预处理所有文本---------------------------------
def multi_process(split_method=None,train_ratio=0.8):
    """
    功能: 对所有文本尽心预处理操作
    param split_method: 切割文本方法
    param train_ratio: 训练集和测试集划分比例
    return
    """
    
    #删除目录
    if os.path.exists('data/prepare/'):
        shutil.rmtree('data/prepare/')
        
    #创建目录
    if not os.path.exists('data/prepare/train/'):
        os.makedirs('data/prepare/train/')
        os.makedirs('data/prepare/test/')

    #获取所有文件名
    idxs = set([file.split('.')[0] for file in os.listdir('data/'+train_dir)])
    idxs = list(idxs)
    
    #随机划分训练集和测试集
    shuffle(idxs)                         #打乱顺序
    index = int(len(idxs)*train_ratio)    #获取训练集的截止下标
    #获取训练集和测试集文件名集合
    train_ids = idxs[:index]
    test_ids = idxs[index:]

    #--------------------------------------------------------------------
    #                               引入多进程
    #--------------------------------------------------------------------
    #线程池方式调用
    import multiprocessing as mp
    num_cpus = mp.cpu_count()           #获取机器CPU的个数
    pool = mp.Pool(num_cpus)
    
    results = []
    #训练集处理
    for idx in train_ids:
        result = pool.apply_async(process_text, args=(idx,split_method,'train'))
        results.append(result)
    #测试集处理
    for idx in test_ids:
        result = pool.apply_async(process_text, args=(idx,split_method,'test'))
        results.append(result)
    #关闭进程池
    pool.close()
    pool.join()
    [r.get for r in results]


#-------------------------------功能:主函数--------------------------------------
if __name__ == '__main__':
    #print(process_text('0',split_method=split_text,split_name='train'))
    
    multi_process(split_text)



