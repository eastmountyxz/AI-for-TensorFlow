# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:39:19 2020
@author: xiuzhang Eastmount CSDN
"""
import os
import glob
import cv2
import numpy as np
import tensorflow as tf

# 定义图片路径
path = 'photo/'

#---------------------------------第一步 读取图像-----------------------------------
def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    fpath = []
    for idx, folder in enumerate(cate):
        # 遍历整个目录判断每个文件是不是符合
        for im in glob.glob(folder + '/*.jpg'):
            #print('reading the images:%s' % (im))
            img = cv2.imread(im)             #调用opencv库读取像素点
            img = cv2.resize(img, (32, 32))  #图像像素大小一致
            imgs.append(img)                 #图像数据
            labels.append(idx)               #图像类标
            fpath.append(path+im)            #图像路径名
            #print(path+im, idx)
    return np.asarray(fpath, np.string_), np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

# 读取图像
fpaths, data, label = read_img(path)
print(data.shape)  # (1000, 256, 256, 3)
# 计算有多少类图片
num_classes = len(set(label))
print(num_classes)

# 生成等差数列随机调整图像顺序
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]
fpaths = fpaths[arr]

# 拆分训练集和测试集 80%训练集 20%测试集
ratio = 0.8
s = np.int(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
fpaths_train = fpaths[:s] 
x_val = data[s:]
y_val = label[s:]
fpaths_test = fpaths[s:] 
print(len(x_train),len(y_train),len(x_val),len(y_val)) #800 800 200 200
print(y_val)


#---------------------------------第二步 建立神经网络-----------------------------------
# 定义Placeholder
xs = tf.placeholder(tf.float32, [None, 32, 32, 3])  #每张图片32*32*3个点
ys = tf.placeholder(tf.int32, [None])               #每个样本有1个输出
# 存放DropOut参数的容器 
drop = tf.placeholder(tf.float32)                   #训练时为0.25 测试时为0

# 定义卷积层 conv0
conv0 = tf.layers.conv2d(xs, 20, 5, activation=tf.nn.relu)    #20个卷积核 卷积核大小为5 Relu激活
# 定义max-pooling层 pool0
pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])        #pooling窗口为2x2 步长为2x2
print("Layer0：\n", conv0, pool0)
 
# 定义卷积层 conv1
conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu) #40个卷积核 卷积核大小为4 Relu激活
# 定义max-pooling层 pool1
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])        #pooling窗口为2x2 步长为2x2
print("Layer1：\n", conv1, pool1)

# 将3维特征转换为1维向量
flatten = tf.layers.flatten(pool1)

# 全连接层 转换为长度为400的特征向量
fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)
print("Layer2：\n", fc)

# 加上DropOut防止过拟合
dropout_fc = tf.layers.dropout(fc, drop)

# 未激活的输出层
logits = tf.layers.dense(dropout_fc, num_classes)
print("Output：\n", logits)

# 定义输出结果
predicted_labels = tf.arg_max(logits, 1)


#---------------------------------第三步 定义损失函数和优化器---------------------------------

# 利用交叉熵定义损失
losses = tf.nn.softmax_cross_entropy_with_logits(
        labels = tf.one_hot(ys, num_classes),       #将input转化为one-hot类型数据输出
        logits = logits)

# 平均损失
mean_loss = tf.reduce_mean(losses)

# 定义优化器 学习效率设置为0.0001
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(losses)


#------------------------------------第四步 模型训练和预测-----------------------------------
# 用于保存和载入模型
saver = tf.train.Saver()
# 训练或预测
train = False
# 模型文件路径
model_path = "model/image_model"

with tf.Session() as sess:
    if train:
        print("训练模式")
        # 训练初始化参数
        sess.run(tf.global_variables_initializer())
        # 定义输入和Label以填充容器 训练时dropout为0.25
        train_feed_dict = {
                xs: x_train,
                ys: y_train,
                drop: 0.25
        }
        # 训练学习1000次
        for step in range(1000):
            _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)
            if step % 20 == 0:  #每隔20次输出一次结果
                # 训练准确率
                pre = sess.run(predicted_labels, feed_dict=train_feed_dict)
                accuracy = 1.0*sum(y_train==pre) / len(pre)
                print("{},{},{}".format(step, mean_loss_val,accuracy))
        # 保存模型
        saver.save(sess, model_path)
        print("训练结束，保存模型到{}".format(model_path))
    else:
        print("测试模式")
        # 测试载入参数
        saver.restore(sess, model_path)
        print("从{}载入模型".format(model_path))
        # label和名称的对照关系
        label_name_dict = {
            0: "人类",
            1: "沙滩",
            2: "建筑",
            3: "公交",
            4: "恐龙",
            5: "大象",
            6: "花朵",
            7: "野马",
            8: "雪山",
            9: "美食"
        }
        # 定义输入和Label以填充容器 测试时dropout为0
        test_feed_dict = {
            xs: x_val,
            ys: y_val,
            drop: 0
        }
        
        # 真实label与模型预测label
        predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
        for fpath, real_label, predicted_label in zip(fpaths_test, y_val, predicted_labels_val):
            # 将label id转换为label名
            real_label_name = label_name_dict[real_label]
            predicted_label_name = label_name_dict[predicted_label]
            print("{}\t{} => {}".format(fpath, real_label_name, predicted_label_name))
        # 评价结果
        print("正确预测个数:", sum(y_val==predicted_labels_val))
        print("准确度为:", 1.0*sum(y_val==predicted_labels_val) / len(y_val))
        k = 0
        while k < len(y_val):
            print(y_val[k], predicted_labels_val[k])
            k = k + 1
            
