# -*- coding: utf-8 -*-
"""
author: Eastmount CSDN 2020-11-15
"""
import os

#评价指标 参数顺序
def classification_pj(pre, y_test):    
    # 正确率 Precision = 正确识别的个体总数 /识别出的个体总数
    # 召回率 Recall = 正确识别的个体总数 /  测试集中存在的个体总数
    # F值 F-measure = 正确率 * 召回率 * 2 / (正确率 + 召回率)

    YC_A, YC_B = 0,0  #预测 bad good
    ZQ_A, ZQ_B = 0,0  #正确
    CZ_A, CZ_B = 0,0  #存在

    #0-good 1-bad 同时计算防止类标变化
    i = 0
    while i<len(pre):
        z = int(y_test[i])   #真实 
        y = int(pre[i])      #预测

        if z==0:
            CZ_A += 1
        elif z==1:
            CZ_B += 1
            
        if y==0:
            YC_A += 1
        elif y==1:
            YC_B += 1
            
        if z==y and z==0 and y==0:
            ZQ_A += 1
        elif z==y and z==1 and y==1:
            ZQ_B += 1
        i = i + 1

    # 结果输出
    print(YC_A, YC_B, ZQ_A, ZQ_B,CZ_A, CZ_B)
    P_A = ZQ_A * 1.0 / (YC_A + 0.1)
    P_B = ZQ_B * 1.0 / (YC_B + 0.1)
    print("Precision 0:{:.4f}".format(P_A))
    print("Precision 1:{:.4f}".format(P_B))
    print("Avg_precision:{:.4f}".format((P_A + P_B)/2))

    R_A = ZQ_A * 1.0 / (CZ_A + 0.1)
    R_B = ZQ_B * 1.0 / (CZ_B + 0.1)
    print("Recall 0:{:.4f}".format(R_A))
    print("Recall 1:{:.4f}".format(R_B))
    print("Avg_recall:{:.4f}".format((R_A + R_B)/2))

    F_A = 2 * P_A * R_A / (P_A + R_A)
    F_B = 2 * P_B * R_B / (P_B + R_B)
    print("F-measure 0:{:.4f}".format(F_A))
    print("F-measure 1:{:.4f}".format(F_B))
    print("Avg_fmeasure:{:.4f}".format((F_A + F_B)/2))
    print("")
