#encoding:utf-8
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
import os
import math
import random

#功能:获取值对应的下标 参数为列表和字符
def item2id(data,w2i):
    #x在字典中直接获取 不在字典中返回UNK
    return [w2i[x] if x in w2i else w2i['UNK'] for x in data]
    
#----------------------------功能:拼接文件---------------------------------
def get_data_with_windows(name='train'):
    #读取prepare_data.py生成的dict.pkl文件 存储字典{类别:下标}
    with open(f'data/dict.pkl', 'rb') as f:
        map_dict = pickle.load(f)   #加载字典
        
    #存储所有数据
    results = []
    root = os.path.join('data/prepare/'+name)
    files = list(os.listdir(root))
    print(files)
    #['10.csv', '11.csv', '12.csv',.....]

    #获取所有文件 进度条
    for file in tqdm(files):
        all_data = []
        path = os.path.join(root, file)
        samples = pd.read_csv(path,sep=',')
        max_num = len(samples)
        #获取sep换行分隔符下标 -1 20 40 60
        sep_index = [-1]+samples[samples['word']=='sep'].index.tolist()+[max_num]
        #print(sep_index)
        #[-1, 83, 92, 117, 134, 158, 173, 200,......]

        #----------------------------------------------------------------------
        #                  获取句子并将句子全部都转换成id
        #----------------------------------------------------------------------
        for i in range(len(sep_index)-1):
            start = sep_index[i] + 1     #0 (-1+1)
            end = sep_index[i+1]         #20
            data = []
            #每个特征进行处理
            for feature in samples.columns:    #访问每列
                #通过函数item2id获取下标 map_dict两个值(列表和字典) 获取第二个值
                data.append(item2id(list(samples[feature])[start:end],map_dict[feature][1]))
            #将每句话的列表合成
            all_data.append(data)

        #----------------------------------------------------------------------
        #                             数据增强
        #----------------------------------------------------------------------
        #前后两个句子拼接 每个句子六个元素(汉字、边界、词性、类别、偏旁、拼音)
        two = []
        for i in range(len(all_data)-1):
            first = all_data[i]
            second = all_data[i+1]
            two.append([first[k]+second[k] for k in range(len(first))]) #六个元素

        three = []
        for i in range(len(all_data)-2):
            first = all_data[i]
            second = all_data[i+1]
            third = all_data[i+2]
            three.append([first[k]+second[k]+third[k] for k in range(len(first))])
            
        #返回所有结果
        results.extend(all_data+two+three)
        
    #return results

    #数据存储至本地 每次调用时间成本过大
    with open(f'data/'+name+'.pkl', 'wb') as f:
        pickle.dump(results, f)
        
#----------------------------功能:批处理---------------------------------
class BatchManager(object):

    def __init__(self, batch_size, name='train'):
        #调用函数拼接文件
        #data = get_data_with_windows(name)
        
        #读取文件
        with open(f'data/'+name+'.pkl', 'rb') as f:
            data = pickle.load(f)
        print(len(data))         #265455句话
        print(len(data[0]))      #6种类别
        print(len(data[0][0]))   #第一句包含字的数量 83
        print("原始数据:", data[0])
                               
        #数据批处理
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        #计算总批次数量 26546
        num_batch = int(math.ceil(len(data) / batch_size))
        #按照句子长度排序
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        
        #获取一个批次的数据
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*int(batch_size) : (i+1)*int(batch_size)]))
        print("分批输出:", batch_data[100])
        
        return batch_data

    @staticmethod
    def pad_data(data_):
        #定义变量
        chars = []
        bounds = []
        flags = []
        radicals = []
        pinyins = []
        targets = []
        
        #print("每个批次句子个数:", len(data_))            #10
        #print("每个句子包含元素个数:", len(data_[0]))     #6
        #print("输出data:", data_)
        
        max_length = max([len(sentence[0]) for sentence in data_])  #值为1
        #print(max_length)
        
        #每个批次共有十组数据 每组数据均为六个元素
        for line in data_:
            #char, bound, flag, target, radical, pinyin = line
            char, target, bound, flag, radical, pinyin = line
            padding = [0] * (max_length - len(char))    #计算补充字符数量
            #注意char和chars不要写错 否则造成递归循环赋值错误
            chars.append(char + padding)
            targets.append(target + padding)
            bounds.append(bound + padding)
            flags.append(flag + padding)
            radicals.append(radical + padding)
            pinyins.append(pinyin + padding)
            
        return [chars, targets, bounds, flags, radicals, pinyins]

    #每次使用一个批次数据
    def iter_batch(self, shuffle=False):
        if shuffle: #乱序
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]
            
#-------------------------------功能:主函数--------------------------------------
if __name__ == '__main__':
    #1.拼接文件(第一次执行 后续可注释)
    #get_data_with_windows('train')

    #2.分批处理 
    train_data = BatchManager(10, 'train')
    
    #3.接着处理下测试集数据
    get_data_with_windows('test')

