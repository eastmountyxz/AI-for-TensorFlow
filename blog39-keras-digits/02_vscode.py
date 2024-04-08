# -*- coding: utf-8 -*- 
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

#导入数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

#One-hot编码
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#构建神经网络模型
model = Sequential()
model.add(Dense(512, input_shape=(28*28,), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax')) #十分类

#训练和预测
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1, validation_split=0.05)
loss, accuracy = model.evaluate(X_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)
