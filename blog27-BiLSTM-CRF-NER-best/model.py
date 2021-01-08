#encoding:utf-8
"""
Created on Thu Jan  7 12:56:40 2021
@author: xiuzhang
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
#计算条件随机场的对数似然
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode

#---------------------------功能：预测计算函数-----------------------------
def network(inputs,shapes,num_entity,lstm_dim=100,
            initializer=tf.truncated_normal_initializer):
    """
    功能：接收一个批次样本的特征数据，计算网络的输出值
    :param char: int, id of chars a tensor of shape 2-D [None,None] 批次数量*每个批次句子长度
    :param bound: int, a tensor of shape 2-D [None,None]
    :param flag: int, a tensor of shape 2-D [None,None]
    :param radical: int, a tensor of shape 2-D [None,None]
    :param pinyin: int, a tensor of shape 2-D [None,None]
    :param shapes: 词向量形状字典
    :param lstm_dim: 神经元的个数
    :param num_entity: 实体标签数量 31种类型
    :param initializer: 初始化函数
    :return
    """
    #--------------------------------------------------
    #特征嵌入:将所有特征的id转换成一个固定长度的向量
    #--------------------------------------------------
    embedding = []
    keys = list(shapes.keys())
    print("Network Input:", inputs)
    #{'char':<tf.Tensor 'char_inputs_10:0' shape=(?, ?) dtype=int32>,
    print("Network Shape:", keys) 
    #['char', 'bound', 'flag', 'radical', 'pinyin']
    
    #循环将五类特征转换成词向量 后续拼接
    for key in keys:   #char
        with tf.variable_scope(key+'_embedding'):
            #获取汉字信息
            lookup = tf.get_variable(
                name = key + '_embedding',         #名称
                shape = shapes[key],               #[num,dim] 行数(字个数)*列数(向量维度) 1663*100
                initializer = initializer
            )
            #词向量映射 汉字结果[None,None,100] 每个字映射100维向量 inputs对应每个字
            embedding.append(tf.nn.embedding_lookup(lookup, inputs[key]))
    print("Network Embedding:", embedding)
    #[<tf.Tensor 'char_embedding_14:0' shape=(?, ?, 100) dtype=float32>,
    
    #拼接词向量 shape[None,None,char_dim+bound_dim+flag_dim+radical_dim+pinyin_dim]
    embed = tf.concat(embedding,axis=-1)  #最后一个维度上拼接 -1
    print("Network Embed:", embed, '\n')
    #Tensor("concat:0", shape=(?, ?, 270), dtype=float32) 
    
    #lengths: 计算输入inputs每句话的实际长度(填充内容不计算)
    #填充值PAD下标为0 因此总长度减去PAD数量即为实际长度 从而提升运算效率
    sign = tf.sign(tf.abs(inputs[keys[0]]))             #char 字符长度
    lengths = tf.reduce_sum(sign, reduction_indices=1)  #第二个维度
    
    #获取填充序列长度 char的第二个维度
    num_time = tf.shape(inputs[keys[0]])[1]
    print(sign, lengths, num_time)
    #Tensor("Sign:0", shape=(?, ?), dtype=int32) 
    #Tensor("Sum:0", shape=(?,), dtype=int32) 
    #Tensor("strided_slice:0", shape=(), dtype=int32)
    
    #--------------------------------------------------
    #循环神经网络编码: 双层双向网络
    #--------------------------------------------------
    #第一层
    with tf.variable_scope('BiLSTM_layer1'):
        lstm_cell = {}
        #第一层前向 后向
        for name in ['forward','backward']:
            with tf.variable_scope(name):           #设置名称
                lstm_cell[name] = rnn.BasicLSTMCell(
                    lstm_dim                        #神经元的个数
                )     
        #BiLSTM 2个LSTM组成(各100个神经元)
        outputs1,finial_states1 = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell['forward'],
            lstm_cell['backward'],
            embed,
            dtype = tf.float32,
            sequence_length = lengths               #序列实际长度(该参数可省略)
        )
    #拼接前向LSTM和后向LSTM输出
    outputs1 = tf.concat(outputs1,axis=-1)  #b,L,2*lstm_dim
    print('Network BiLSTM-1:', outputs1)
    #Tensor("concat_1:0", shape=(?, ?, 200), dtype=float32)
    
    #第二层
    with tf.variable_scope('BiLSTM_layer2'):
        lstm_cell = {}
        #第一层前向 后向
        for name in ['forward','backward']:
            with tf.variable_scope(name):           #设置名称
                lstm_cell[name] = rnn.BasicLSTMCell(
                    lstm_dim                        #神经元的个数
                )
        #BiLSTM
        outputs,finial_states = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell['forward'],
            lstm_cell['backward'],
            outputs1,                                #是否利用第一层网络
            dtype = tf.float32,
            sequence_length = lengths                #序列实际长度(该参数可省略)
        )
    #最终结果 [batch_size,maxlength,2*lstm_dim] 即200
    result = tf.concat(outputs,axis=-1)
    print('Network BiLSTM-2:', result)
    #Tensor("concat_2:0", shape=(?, ?, 200), dtype=float32)
    
    #--------------------------------------------------
    #输出全连接映射
    #--------------------------------------------------
    #转换成二维矩阵再进行乘法操作 [batch_size*maxlength,2*lstm_dim]
    result = tf.reshape(result, [-1,2*lstm_dim])
    
    #第一层映射 矩阵乘法 200映射到100
    with tf.variable_scope('project_layer1'):
        #权重
        w = tf.get_variable(
            name = 'w',
            shape = [2*lstm_dim,lstm_dim],     #转100维
            initializer = initializer
        )
        #bias
        b = tf.get_variable(
            name = 'b',
            shape = [lstm_dim],
            initializer = tf.zeros_initializer()
        )
        #运算 激活函数relu
        result = tf.nn.relu(tf.matmul(result,w)+b)
    print("Dense-1:",result)
    #Tensor("project_layer1/Relu:0", shape=(?, 100), dtype=float32)
    
    #第二层映射 矩阵乘法 100映射到31
    with tf.variable_scope('project_layer2'):
        #权重
        w = tf.get_variable(
            name = 'w',
            shape = [lstm_dim,num_entity],     #31种实体类别
            initializer = initializer
        )
        #bias
        b = tf.get_variable(
            name = 'b',
            shape = [num_entity],
            initializer = tf.zeros_initializer()
        )
        #运算 激活函数relu 最后一层不激活
        result = tf.matmul(result,w)+b
    print("Dense-2:",result)
    #Tensor("project_layer2/add:0", shape=(?, 31), dtype=float32)
    
    #形状转换成三维
    result = tf.reshape(result, [-1,num_time,num_entity])
    print('Result:', result, "\n")
    #Tensor("Reshape_1:0", shape=(?, ?, 31), dtype=float32)
    
    #[batch_size,max_length,num_entity]
    return result,lengths

#-----------------------------功能：定义模型类---------------------------
class Model(object):
    
    #---------------------------------------------------------
    #初始化
    def __init__(self, dict_, lr=0.0001):
        #通过dict.pkl计算各个特征数量
        self.num_char = len(dict_['word'][0])
        self.num_bound = len(dict_['bound'][0])
        self.num_flag = len(dict_['flag'][0])
        self.num_radical = len(dict_['radical'][0])
        self.num_pinyin = len(dict_['pinyin'][0])
        self.num_entity = len(dict_['label'][0])
        print('model init:', self.num_char, self.num_bound, self.num_flag,
              self.num_radical, self.num_pinyin, self.num_entity)
        
        #字符映射成向量的维度
        self.char_dim = 100
        self.bound_dim = 20
        self.flag_dim = 50
        self.radical_dim = 50
        self.pinyin_dim = 50
        
        #shape表示为[num,dim] 行数(个数)*列数(向量维度)
        
        #设置LSTM的维度 神经元的个数
        self.lstm_dim = 100
        
        #学习率
        self.lr = lr
        
        #保存初始化字典
        self.map = dict_
      
        #---------------------------------------------------------
        #定义接收数据的placeholder [None,None] 批次 句子长度
        self.char_inputs = tf.placeholder(dtype=tf.int32,shape=[None,None],name='char_inputs')
        self.bound_inputs = tf.placeholder(dtype=tf.int32,shape=[None,None],name='bound_inputs')
        self.flag_inputs = tf.placeholder(dtype=tf.int32,shape=[None,None],name='flag_inputs')
        self.radical_inputs = tf.placeholder(dtype=tf.int32,shape=[None,None],name='radical_inputs')
        self.pinyin_inputs = tf.placeholder(dtype=tf.int32,shape=[None,None],name='pinyin_inputs')
        self.targets = tf.placeholder(dtype=tf.int32,shape=[None,None],name='targets')    #目标真实值
        self.global_step = tf.Variable(0,trainable=False)  #不能训练 用于计数
                
        #---------------------------------------------------------
        #传递给网络 计算模型输出值
        #参数：输入的字、边界、词性、偏旁、拼音下标 -> network转换词向量并计算
        #返回：网络输出值、每句话的真实长度
        self.logits,self.lengths = self.get_logits(
            self.char_inputs,
            self.bound_inputs,
            self.flag_inputs,
            self.radical_inputs,
            self.pinyin_inputs
        )
        
        #---------------------------------------------------------
        #计算损失 
        #参数：模型输出值、真实标签序列、长度(不计算填充)
        #返回：损失值
        self.cost = self.loss(
            self.logits,
            self.targets,
            self.lengths
        )
        print("Cost:", self.cost)
        
        #---------------------------------------------------------
        #优化器优化 采用梯度截断技术
        with tf.variable_scope('optimizer'):
            opt = tf.train.AdamOptimizer(self.lr)      #学习率
            #计算所有损失函数的导数值
            grad_vars = opt.compute_gradients(self.cost)
            #梯度截断-导数值过大会导致步子迈得过大 梯度爆炸(因此限制在某个范围内)
            #grad_vars记录每组参数导数和本身
            clip_grad_vars = [[tf.clip_by_value(g,-5,5),v] for g,v in grad_vars]
            #使用截断后的梯度更新参数 该方法每应用一次global_step参数自动加1
            self.train_op = opt.apply_gradients(clip_grad_vars, self.global_step)
            print("Optimizer:", self.train_op)
            
        #模型保存 保留最近5次模型
        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=5)
        
    #---------------------------------------------------------
    #定义网络 接收批次样本
    def get_logits(self,char,bound,flag,radical,pinyin): 
        """
        功能：接收一个批次样本的特征数据，计算网络的输出值
        :param char: int, id of chars a tensor of shape 2-D [None,None]
        :param bound: int, a tensor of shape 2-D [None,None]
        :param flag: int, a tensor of shape 2-D [None,None]
        :param radical: int, a tensor of shape 2-D [None,None]
        :param pinyin: int, a tensor of shape 2-D [None,None]
        :return: 返回3-d tensor [batch_size,max_length,num_entity]
        """
        #定义字典传参
        shapes = {}
        shapes['char'] = [self.num_char,self.char_dim]
        shapes['bound'] = [self.num_bound,self.bound_dim]
        shapes['flag'] = [self.num_flag,self.flag_dim]
        shapes['radical'] = [self.num_radical,self.radical_dim]
        shapes['pinyin'] = [self.num_pinyin,self.pinyin_dim]
        print("shapes:", shapes, '\n')
        #{'char': [1663, 100], 'bound': [5, 20], 'flag': [56, 50], 
        # 'radical': [227, 50], 'pinyin': [989, 50]}        
        
        #输入参数定义字典
        inputs = {}
        inputs['char'] = char
        inputs['bound'] = bound
        inputs['flag'] = flag
        inputs['radical'] = radical
        inputs['pinyin'] = pinyin
        
        #return network(char,bound,flag,radical,pinyin,shapes)
        return network(inputs,shapes,lstm_dim=self.lstm_dim,num_entity=self.num_entity)

    #--------------------------功能：定义loss CRF模型-------------------------
    #参数: 模型输出值 真实标签序列 长度(不计算填充)
    def loss(self,result,targets,lengths):
        #获取长度
        b = tf.shape(lengths)[0]              #真实长度 该值只有一维
        num_steps = tf.shape(result)[1]       #含填充
        print("Loss lengths:", b, num_steps)
        print("Loss Inputs:", result)
        print("Loss Targets:", targets)
        
        #转移矩阵
        with tf.variable_scope('crf_loss'):
            #取log相当于概率接近0
            small = -1000.0
            
            #初始时刻状态
            start_logits = tf.concat(
                #前31个-1000概率为0 最后一个start为0取log为1
                [small*tf.ones(shape=[b,1,self.num_entity]),tf.zeros(shape=[b,1,1])],
                axis = -1   #两个矩阵在最后一个维度合并
            )
            
            #X值拼接 每个时刻加一个状态
            pad_logits = tf.cast(small*tf.ones([b,num_steps,1]),tf.float32)
            logits = tf.concat([result, pad_logits], axis=-1)
            logits = tf.concat([start_logits,logits], axis=1) #第二个位置拼接
            print("Loss Logits:", logits)
            
            #Y值拼接
            targets = tf.concat(
                [tf.cast(self.num_entity*tf.ones([b,1]),tf.int32),targets],
                axis = -1
            )
            print("Loss Targets:", targets)
            
            #计算
            self.trans = tf.get_variable(
                name = 'trans',
                #初始概率start加1 最终32个
                shape = [self.num_entity+1,self.num_entity+1],
                initializer = tf.truncated_normal_initializer()
            )
            
            #损失 计算条件随机场的对数似然 每个样本计算几个值
            log_likehood, self.trans = crf_log_likelihood(
                inputs = logits,                   #输入
                tag_indices = targets,             #目标
                transition_params = self.trans,
                sequence_lengths = lengths         #真实样本长度
            )
            print("Loss loglikehood:", log_likehood)
            print("Loss Trans:", self.trans)
            
            #返回所有样本平均值 数加个负号损失最小化
            return tf.reduce_mean(-log_likehood)
       
    #--------------------------功能：分步运行-------------------------
    #参数: 会话、分批数据、训练预测
    def run_step(self,sess,batch,is_train=True):
        if is_train:
            feed_dict = {
                self.char_inputs : batch[0],
                self.bound_inputs : batch[2],
                self.flag_inputs : batch[3],
                self.radical_inputs : batch[4],
                self.pinyin_inputs : batch[5],
                self.targets : batch[1]  #注意顺序
            }
            #训练计算损失
            _,loss = sess.run([self.train_op,self.cost], feed_dict=feed_dict)
            return loss
        else: #预测没有类标
            feed_dict = {
                self.char_inputs : batch[0],
                self.bound_inputs : batch[2],
                self.flag_inputs : batch[3],
                self.radical_inputs : batch[4],
                self.pinyin_inputs : batch[5],
            }
            #测试计算结果
            logits,lengths = sess.run([self.logits, self.lengths], feed_dict=feed_dict)
            return logits,lengths
    
    #--------------------------功能：解码获取id-------------------------
    #参数:模型输出值、真实长度、转移矩阵(用于解码)
    def decode(self,logits,lengths,matrix):
        #保留概率最大路径
        paths = []
        small = -1000.0
        #每个样本解码 31种类别+最后一个是0
        start = np.asarray([[small]*self.num_entity+[0]])
        
        #获取每句话的成绩和样本真实长度
        for score,length in zip(logits,lengths):
            score = score[:length]   #只取有效字符的输出
            pad = small*np.ones([length,1])
            #拼接
            logits = np.concatenate([score,pad],axis=-1)
            logits = np.concatenate([start,logits],axis=0)
            #解码
            path,_ = viterbi_decode(logits,matrix)
            paths.append(path[1:])
        
        #paths获取的是id 还需要转换成对应的实体标签
        return paths
        
    #--------------------------功能：预测分析-------------------------
    #参数: 会话、批次 
    def predict(self,sess,batch):
        results = []
        #获取转移矩阵
        matrix = self.trans.eval()
        
        #获取模型结果 执行测试
        logits, lengths = self.run_step(sess, batch, is_train=False)
        
        #调用解码函数获取paths
        paths = self.decode(logits, lengths, matrix)
        
        #查看字及对应的标记
        chars = batch[0]
        for i in range(len(paths)):  #有多少路径就有多少句子
            #获取第i句话真实长度
            length = lengths[i]
            #第i句话真实的字
            chars[i][:length]
            #ID转换成对应的每个字
            #map['word'][1]是字典
            string = [self.map['word'][1][index] for index in chars[i][:length]]
            #获取tag
            tags = [self.map['label'][0][index] for index in paths[i]]
            #形成完整列表
            result = [k for c,t in zip(string,tags)]
            results.append(result)
            
        #获取预测值
        return results