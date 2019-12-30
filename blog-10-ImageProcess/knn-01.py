# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:37:10 2019
@author: xiuzhang
"""
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#----------------------------------------------------------------------------------
# 第一步 切分训练集和测试集
#----------------------------------------------------------------------------------

X = [] #定义图像名称
Y = [] #定义图像分类类标
Z = [] #定义图像像素

for i in range(0, 3):
    #遍历文件夹，读取图片
    for f in os.listdir("data/%s" % i):
        #获取图像名称
        X.append("data//" +str(i) + "//" + str(f))
        #获取图像类标即为文件夹名称
        Y.append(i)
        #print("data//" +str(i) + "//" + str(f))

X = np.array(X)
Y = np.array(Y)

#随机率为100% 选取其中的30%作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.3, random_state=1)

print(len(X_train), len(X_test), len(y_train), len(y_test))

#----------------------------------------------------------------------------------
# 第二步 图像读取及转换为像素直方图
#----------------------------------------------------------------------------------

#训练集
XX_train = []
for i in X_train:
    #读取图像
    #print(i)
    image = cv2.imread(i)
    
    #图像像素大小一致
    #print(image.shape)
    img = cv2.resize(image, (32,32))

    #计算图像直方图并存储至X数组
    hist = cv2.calcHist([img], [0,1], None,
                            [256,256], [0.0,255.0,0.0,255.0])

    XX_train.append(((hist/255).flatten()))

#测试集
XX_test = []
for i in X_test:
    #读取图像
    #print(i)
    image = cv2.imread(i)
    
    #图像像素大小一致
    img = cv2.resize(image, (256,256),
                     interpolation=cv2.INTER_CUBIC)

    #计算图像直方图并存储至X数组
    hist = cv2.calcHist([img], [0,1], None,
                            [256,256], [0.0,255.0,0.0,255.0])

    XX_test.append(((hist/255).flatten()))

#----------------------------------------------------------------------------------
# 第三步 基于KNN的图像分类处理
#----------------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=11).fit(XX_train, y_train)
predictions_labels = clf.predict(XX_test)

print('预测结果:')
print(predictions_labels)

print('算法评价:')
print(classification_report(y_test, predictions_labels))

#输出前10张图片及预测结果
k = 0
while k<10:
    #读取图像
    print(X_test[k])
    image = cv2.imread(X_test[k])
    print(predictions_labels[k])
    #显示图像
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    k = k + 1


"""
700 300 700 300
预测结果:
[5 0 4 3 5 9 2 6 3 0 4 9 5 3 0 9 5 0 6 4 7 4 5 9 4 5 5 6 5 3 9 7 5 4 4 2 0
 7 6 6 0 7 0 9 0 0 5 0 4 5 7 0 6 0 5 4 9 9 4 6 9 0 3 7 8 9 4 0 5 0 0 5 5 0
 4 0 5 5 3 6 5 7 7 7 0 9 4 9 5 5 7 9 0 9 4 6 0 5 3 5 3 0 9 5 4 9 0 5 7 9 4
 0 0 0 0 0 7 4 5 7 5 9 0 5 4 7 7 7 6 7 0 0 6 0 0 0 7 9 3 7 0 5 9 7 9 7 7 5
 5 5 0 9 0 0 0 7 7 6 0 2 5 5 5 6 0 4 0 5 7 5 5 0 4 4 5 6 5 0 0 9 0 0 7 6 5
 0 9 0 5 6 9 0 2 5 9 5 7 5 6 0 3 6 3 0 5 6 0 3 3 0 6 3 9 4 9 3 3 5 9 9 3 0
 3 9 5 7 5 7 9 0 0 6 9 4 7 9 0 7 0 0 6 0 3 6 7 3 0 4 9 7 5 0 7 6 9 6 9 0 3
 0 9 5 5 0 7 5 3 5 4 5 4 0 5 0 9 7 9 5 3 3 7 0 0 0 0 0 0 0 4 9 6 5 7 5 5 4
 0 0 0 5]

算法评价:
              precision    recall  f1-score   support

           0       0.38      0.97      0.55        31
           1       0.00      0.00      0.00        31
           2       1.00      0.15      0.27        26
           3       0.83      0.69      0.75        29
           4       1.00      0.88      0.93        32
           5       0.35      0.62      0.45        34
           6       0.81      0.70      0.75        30
           7       0.61      0.88      0.72        26
           8       1.00      0.03      0.06        31
           9       0.44      0.60      0.51        30

    accuracy                           0.55       300
   macro avg       0.64      0.55      0.50       300
weighted avg       0.63      0.55      0.50       300

photo//5//507.jpg
5
photo//8//818.jpg
0
photo//4//452.jpg
4
"""
