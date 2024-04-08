# -*- coding: utf-8 -*- 
from keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
for i in range(4):
   plt.subplot(2,2,i+1)
   plt.imshow(X_train[i], cmap=plt.get_cmap('gray'), interpolation='none')
   plt.title("Class {}".format(y_train[i]))
plt.show()
