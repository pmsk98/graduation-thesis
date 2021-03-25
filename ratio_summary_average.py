# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 16:12:31 2021

@author: user
"""



####기준 라벨링 diff

import glob
import os
import pandas as pd
import  talib
import numpy as np
path='C:/Users/user/Desktop/수수료 적용 ratio/5_기간 변동성/표준편차0.25'

file_list =os.listdir(path)

df= []

for file in file_list:
    path = 'C:/Users/user/Desktop/수수료 적용 ratio/5_기간 변동성/표준편차0.25'
    df.append(pd.read_csv(path+"/"+file,encoding='euc-kr'))
    
    
#drop => Unnamed: 0,year,stock_name,No.trades

#나머지 지표 평균
avg_ratio=[]
pr_1=[]

for i in range(len(df)):
    pr_1.append(df[i].groupby(by=['stock_name']).mean())
    pr_1[i]=pr_1[i].drop(['Unnamed: 0','year'],axis=1)
    avg_ratio.append(pr_1[i])
    


#리스트 -> 데이터프레임 변환

for i in range(len(df)):
    avg_ratio[i]=pd.DataFrame(avg_ratio[i])




#지표 평균
for i in range(len(df)):
    avg_ratio[i].loc[len(avg_ratio[i])] =avg_ratio[i].mean()
    avg_ratio[i].rename(index={73:'total_average'},inplace=True)






len(method_name)
method_name=['method1_decsion','method1_gbm','method1_knn','method1_logistic','method1_naive','method1_neural','method1_randomforest','method1_svm','method1_voting',
             'method2_decsion','method2_gbm','method2_knn','method2_logistic','method2_naive','method2_neural','method2_randomforest','method2_svm','method2_voting']





for i in range(len(df)):
    avg_ratio[i].to_csv('{}_summary_0.25.csv'.format(method_name[i]),encoding='euc-kr')






