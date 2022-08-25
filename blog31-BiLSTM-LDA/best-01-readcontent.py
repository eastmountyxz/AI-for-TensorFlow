# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:21:53 2021
@author: xiuzhang
"""
import pandas as pd

pd_all = pd.read_csv('simplifyweibo_4_moods.csv')
moods = {0: '喜悦', 1: '愤怒', 2: '厌恶', 3: '低落'}

print('微博数目（总体）：%d' % pd_all.shape[0])
for label, mood in moods.items(): 
    print('微博数目（{}）：{}'.format(mood,pd_all[pd_all.label==label].shape[0]))
    

