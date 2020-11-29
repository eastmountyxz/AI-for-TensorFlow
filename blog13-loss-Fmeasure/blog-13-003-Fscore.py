# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:39:19 2020
@author: xiuzhang Eastmount CSDN
"""
import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------
# 第一部分 计算准确率 召回率 F值
#--------------------------------------------------------------------------

# 读取文件数据
fp = open('test_data.txt', 'r')

# 迭代次数 整体误差 正确率
real = []
pre = []

# 解析数据
for line in fp.readlines():
    con = line.strip('\n').split(' ')
    #print(con)
    real.append(int(con[0])) #真实类标
    pre.append(int(con[1]))  #预测类标

# 计算各类结果 共10类图片
real_10 = list(range(0, 10))   #真实10个类标数量的统计
pre_10 = list(range(0, 10))    #预测10个类标数量的统计
right_10 = list(range(0, 10))  #预测正确的10个类标数量

k = 0
while k < len(real):
    v1 = int(real[k])
    v2 = int(pre[k])
    print(v1, v2)
    real_10[v1] = real_10[v1] + 1     # 计数
    pre_10[v2] = pre_10[v2] + 1       # 计数
    if v1==v2:
        right_10[v1] = right_10[v1] + 1
    k = k + 1
print("统计各类数量")
print(real_10, pre_10, right_10)

# 准确率 = 正确数 / 预测数
precision = list(range(0, 10))
k = 0
while k < len(real_10):
    value = right_10[k] * 1.0 / pre_10[k] 
    precision[k] = value
    k = k + 1
print(precision)

# 召回率 = 正确数 / 真实数
recall = list(range(0, 10))
k = 0
while k < len(real_10):
    value = right_10[k] * 1.0 / real_10[k] 
    recall[k] = value
    k = k + 1
print(recall)
   
# F值 = 2*准确率*召回率/(准确率+召回率)
f_measure = list(range(0, 10))
k = 0
while k < len(real_10):
    value = (2 * precision[k] * recall[k] * 1.0) / (precision[k] + recall[k])
    f_measure[k] = value
    k = k + 1
print(f_measure)

#--------------------------------------------------------------------------
# 第二部分 绘制曲线
#--------------------------------------------------------------------------

# 设置类别
n_groups = 10
fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.2
 
opacity = 0.4
error_config = {'ecolor': '0.3'}

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
 
# 绘制
rects1 = ax.bar(index, precision, bar_width,
                alpha=opacity, color='b',
                error_kw=error_config,
                label='precision')
 
rects2 = ax.bar(index + bar_width, recall, bar_width,
                alpha=opacity, color='m',
                error_kw=error_config,
                label='recall')
 
rects3 = ax.bar(index + bar_width + bar_width, f_measure, bar_width,
                alpha=opacity, color='r',
                error_kw=error_config,
                label='f_measure')
            
# 设置标签
ax.set_xticks(index + 3 * bar_width / 3)
ax.set_xticklabels(('0-人类', '1-沙滩', '2-建筑', '3-公交', '4-恐龙',
                    '5-大象', '6-花朵', '7-野马', '8-雪山', '9-美食'))
# 设置类标
ax.legend()
plt.xlabel("类标")
plt.ylabel("评价")
fig.tight_layout()
plt.savefig('result.png', dpi=200)
plt.show()

