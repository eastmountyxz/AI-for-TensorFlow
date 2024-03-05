#encoding:utf-8
# By: Eastmount 2024-02-15
# 参考：https://www.bilibili.com/video/BV1KZ4y1z7Bx （每天都要机器学习）
# 版本：python 3.7, tf 2.2.0,  keras 2.3.1, bert4keras 0.11.5
import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.layers import ConditionalRandomField
from keras.layers import Embedding, Bidirectional, LSTM, Dense, \
     TimeDistributed, Dropout, Input, Activation, concatenate

#------------------------------------------------------
#搭建模型
#参数：Bert配置文件路径 LSTM单元数 dropout参数
#------------------------------------------------------
def bert_bilstm_crf(config_path,checkpoint_path,num_labels,
                    lstm_units,drop_rate,learning_rate):
    #定义bert模型 使用bert自带输入
    bert = build_transformer_model(
            config_path = config_path,
            checkpoint_path = checkpoint_path,
            model = 'bert',
            return_keras_model = False
        )
    #输出Bert编码的三维Embedding向量 每个Tokenizer都有一个768维词向量
    x = bert.model.output  #[batch_size, seq_length, 768]

    #词向量传给BiLSTM模型 初始化及返回每个Tokenizer的输出
    lstm = Bidirectional(
            LSTM(
                lstm_units,
                kernel_initializer = 'he_normal',
                return_sequences = True
            )
        )(x)               #[batch_size, seq_length, lstm_units*2]

    #BiLSTM和Bert输出拼接
    x = concatenate(
        [lstm,x],
        axis=-1            #在-1维度拼接(最后一维)
        )                  #[batch_size, seq_length, lstm_units*2+768]
    
    #序列数据捕获时间信息
    x = TimeDistributed(Dropout(drop_rate))(x)

    #全连接层 标签数量
    x = TimeDistributed(
            Dense(
                num_labels,
                activation='relu',
                kernel_initializer = 'he_normal'
            )
        )(x)               #[batch_size, seq_length, num_labels]

    #构建CRF模型
    crf = ConditionalRandomField()
    output = crf(x)

    #模型实例化及编译
    model = keras.models.Model(bert.input, output)
    model.compile(
            loss = crf.sparse_loss,
            optimizer = Adam(learning_rate),
            metrics = [crf.sparse_accuracy]
        )

    return model,crf

#------------------------------------------------------
#主函数
#------------------------------------------------------
if __name__ == '__main__':
    config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
    lstm_units = 128
    num_labels = 7
    drop_rate = 0.1
    learning_rate = 0.001

    model,crf = bert_bilstm_crf(config_path,checkpoint_path,num_labels,
                                lstm_units,drop_rate,learning_rate)

    #打印模型看到Bert输出的768维向量
    print(model.summary())

    
    
    
