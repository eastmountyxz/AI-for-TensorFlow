# -*- coding: utf-8 -*-
# By：Eastmount CSDN 2022-08-23
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

#-------------------------下载MNIST数据--------------------------------
#只使用x数据集
(x_train, _), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape, _.shape, y_test.shape)

#---------------------------数据预处理--------------------------------
#minmax_normalized 处理至(-0.5,0.5)区间
x_train = x_train.astype('float32') / 255. - 0.5 
x_test = x_test.astype('float32') / 255. - 0.5
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_train.shape, x_test.shape)

#-----------------------构建Encoder和Decoder层-----------------------
#降维可视化绘制2D图
encoding_dim = 2

#input placeholder 28*28
input_img = Input(shape=(784,))

#Encoder layers（压缩）
#利用Dense构造Encoder层，其输出值为128，输入值为input_img
encoded = Dense(128, activation='relu')(input_img)
#第二层的输出是64，输入是上一个构建的encoded
encoded = Dense(64, activation='relu')(encoded)
#第三层压缩至10
encoded = Dense(10, activation='relu')(encoded)
#最后构建需要的自编码压缩器，压缩成2个值，它能代表整个784个特征
encoder_output = Dense(encoding_dim,)(encoded)

#Decoder Layers（解压）
#通常Encoder怎么构建，Decoder也对应反向构建，实现解压处理，重构至784个特征
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
#由于输入值是(-0.5,0.5)，而使用tanh激活函数的范围是(-1,1)，因此实现对应效果
decoded = Dense(784, activation='tanh')(decoded)

#--------------------------构造自编码器模型---------------------------
autoencoder = Model(inputs=input_img, outputs=decoded)

#构建encoder模型进行可视化分析
encoder = Model(inputs=input_img, outputs=encoder_output)

#激活自编码器
autoencoder.compile(optimizer='adam', loss='mse')

#-----------------------------训练和测试------------------------------
#输入和输出均是x_train,对比二者形成误差
autoencoder.fit(x_train, 
                x_train, 
                epochs=20,
                batch_size=256,
                shuffle=True)

#预测
encoded_imgs = encoder.predict(x_test)     #压缩二维特征 用于聚类
decoded_imgs = autoencoder.predict(x_test) #自编码器还原的图像

#比较原始图像和预测图像数据
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    a[0][i].imshow(np.reshape(x_test[i], (28, 28)))
    a[1][i].imshow(np.reshape(decoded_imgs[i], (28, 28)))
plt.show()

#聚类分析
plt.scatter(encoded_imgs[:,0], encoded_imgs[:,1], c=y_test)
plt.colorbar()
plt.show()


