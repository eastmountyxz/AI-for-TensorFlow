# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 18:57:23 2021
@author: xiuzhang
"""
import tensorflow as tf
from data_utils import BatchManager
import pickle
from model import Model
import time

#-----------------------------功能：读取字典---------------------------
dict_file = 'data/dict.pkl'
def get_dict(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

#-----------------------------功能：训练函数---------------------------
batch_size = 20
def train():
    #调用已定义的方法获取处理好的数据集
    train_manager = BatchManager(batch_size=20, name='train')
    print('train:', type(train_manager))    #<class 'data_utils.BatchManager'>
    test_manager = BatchManager(batch_size=100, name='test')
    
    #读取字典
    mapping_dict = get_dict(dict_file)
    print('train:', len(mapping_dict))   #6
    print('计算六元组个数')
    print('字:', len(mapping_dict['word'][0]))              #1663
    print('边界:', len(mapping_dict['bound'][0]))           #5
    print('词性:', len(mapping_dict['flag'][0]))            #56
    print('偏旁:', len(mapping_dict['radical'][0]))         #227
    print('拼音:', len(mapping_dict['pinyin'][0]))          #989
    print('类别:', len(mapping_dict['label'][0]),'\n')      #31
    
    #-------------------------搭建模型---------------------------
    #实例化模型 执行init初始化方法model核心函数：
    #    1.get_logits：传递给网络 计算模型输出值 
    #    2.loss：计算损失值
    #-----------------------------------------------------------
    model = Model(mapping_dict)
    print("---------------模型构建成功---------------------\n")
    
    #初始化训练
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(10):
            j = 1
            #调用iter_batch函数 迭代过程可以让梯度下降在不断尝试找到最优解
            for batch in train_manager.iter_batch(shuffle=True):      #乱序
                #时间计算
                start = time.time()
                #调用自定义函数
                loss = model.run_step(sess,batch)
                end = time.time()
                
                #每10批输出
                if j % 10==0:
                    #第几轮 每批数量 多少批次 损失 消耗时间 剩余估计时间
                    print('epoch:{},step:{}/{},loss:{},elapse:{},estimate:{}'.format(
                            i+1,j,train_manager.len_data,
                            loss,(end-start),
                            (end-start)*(train_manager.len_data-j)))
                j += 1
                
                """
                #print(len(batch))       #6个类型
                #print(len(batch[0]),len(batch[1]),len(batch[2]))     #20个                   
                #每次获取一个批次的数据 feed_dict喂数据 placeholder用于接收神经网络数据
                _,loss = sess.run([model.train_op,model.cost],feed_dict={
                                            model.char_inputs : batch[0],
                                            model.bound_inputs : batch[2],
                                            model.flag_inputs : batch[3],
                                            model.radical_inputs : batch[4],
                                            model.pinyin_inputs : batch[5],
                                            model.targets : batch[1]  #注意顺序
                                            })
                print('loss:{}'.format(loss))
                #InvalidArgumentError: indices[0,2] = 7 is not in [0, 5)
                #注意:feed_dict对应数据必须一致,最早CSV文件label为第2列,所有文件写返回值顺序一致
                #data_utils.py: char, target, bound, flag, radical, pinyin = line
                """
            
            #--------------------------------------------------
            #每迭代一轮进行预测
            for batch in test_manager.iter_batch(shuffle=True):
                print(model.predict(sess,batch))
            
#----------------------------功能:主函数---------------------------------
if __name__ == '__main__':
    train()
    
    