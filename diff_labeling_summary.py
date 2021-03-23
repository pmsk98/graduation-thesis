# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:57:06 2021

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:59:23 2021

@author: user
"""


####기준 라벨링 diff

import glob
import os
import pandas as pd
import  talib
import numpy as np
path='C:/Users/user/Desktop/ratio_eda/diff_label'

file_list =os.listdir(path)

df= []

for file in file_list:
    path = 'C:/Users/user/Desktop/ratio_eda/diff_label'
    df.append(pd.read_csv(path+"/"+file,encoding='euc-kr'))
    
    
    
df[1].columns


#트레이딩 횟수 합계    
sum_trade=[]
pr=[]
for i in range(len(df)):
    
    pr.append(df[i].groupby(by=['stock_name']).sum('No.trades'))
    sum_trade.append(pr[i]['No.trades'])



df[1].groupby(by=['stock_name']).mean()

#drop => Unnamed: 0,year,stock_name,No.trades

#나머지 지표 평균
avg_ratio=[]
pr_1=[]

for i in range(len(df)):
    pr_1.append(df[i].groupby(by=['stock_name']).mean())
    pr_1[i]=pr_1[i].drop(['Unnamed: 0','year','No.trades'],axis=1)
    avg_ratio.append(pr_1[i])
    


#리스트 -> 데이터프레임 변환

for i in range(len(df)):
    avg_ratio[i]=pd.DataFrame(avg_ratio[i])
    sum_trade[i]=pd.DataFrame(sum_trade[i])




#result
method1_result=[]

for i in range(len(df)):
    avg_ratio[i]['No.trades'] =sum_trade[i]['No.trades']
    method1_result.append(avg_ratio[i])



len(method_name)
method_name=['method1_decsion','method1_gbm','method1_knn','method1_logistic','method1_naive','method1_neural','method1_randomforest','method1_svm','method1_voting',
             'method2_decsion','method2_gbm','method2_knn','method2_logistic','method2_naive','method2_neural','method2_randomforest','method2_svm','method2_voting']


for i in range(len(df)):
    method1_result[i].to_csv('{}_summary.csv'.format(method_name[i]),encoding='euc-kr')






