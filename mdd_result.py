# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:57:21 2021

@author: user
"""



#연도별 mdd

import glob
import os
import pandas as pd
import  talib
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

path = "C:/Users/user/Desktop/연구/시가총액100위"

file_list =os.listdir(path)

df= []

for file in file_list:
    path = "C:/Users/user/Desktop/연구/시가총액100위"
    
    df.append(pd.read_csv(path+"/"+file))
    
    
    


stock_df =pd.DataFrame(file_list)

stock_name = stock_df[0].str.split('.csv',expand=True)

#stock_name
stock_name[0]




#2016
test_2016=[]


for i in range(0,73):
    test=None    
    test=df[i]['Date'].str.contains('2016')
    test_2016.append(df[i][test])



mdd_2016=[]


historical_max=[]
daily_drawdown=[]
mdd_1=[]
for i in range(0,73):
    historical_max.append(test_2016[i]['Close'].cummax())
    daily_drawdown.append(test_2016[i]['Close'] / historical_max[i] - 1.0)
    mdd_1.append(daily_drawdown[i].cummin())
    mdd_2016.append(mdd_1[i].min())
    mdd_2016[i]= -1 * mdd_2016[i] *100 # %로 변환



mdd_2016 #2016년도 mdd
    



#2017
test_2017=[]


for i in range(0,73):
    test=None    
    test=df[i]['Date'].str.contains('2017')
    test_2017.append(df[i][test])



mdd_2017=[]


historical_max=[]
daily_drawdown=[]
mdd_1=[]


for i in range(0,73):
    historical_max.append(test_2017[i]['Close'].cummax())
    daily_drawdown.append(test_2017[i]['Close'] / historical_max[i] - 1.0)
    mdd_1.append(daily_drawdown[i].cummin())
    mdd_2017.append(mdd_1[i].min())
    mdd_2017[i]= -1 * mdd_2017[i] *100 # %로 변환




#######2018
test_2018=[]


for i in range(0,73):
    test=None    
    test=df[i]['Date'].str.contains('2018')
    test_2018.append(df[i][test])



mdd_2018=[]


historical_max=[]
daily_drawdown=[]
mdd_1=[]


for i in range(0,73):
    historical_max.append(test_2018[i]['Close'].cummax())
    daily_drawdown.append(test_2018[i]['Close'] / historical_max[i] - 1.0)
    mdd_1.append(daily_drawdown[i].cummin())
    mdd_2018.append(mdd_1[i].min())
    mdd_2018[i]= -1 * mdd_2018[i] *100 # %로 변환
    
    


###2019
test_2019=[]


for i in range(0,73):
    test=None    
    test=df[i]['Date'].str.contains('2019')
    test_2019.append(df[i][test])



mdd_2019=[]


historical_max=[]
daily_drawdown=[]
mdd_1=[]


for i in range(0,73):
    historical_max.append(test_2019[i]['Close'].cummax())
    daily_drawdown.append(test_2019[i]['Close'] / historical_max[i] - 1.0)
    mdd_1.append(daily_drawdown[i].cummin())
    mdd_2019.append(mdd_1[i].min())
    mdd_2019[i]= -1 * mdd_2019[i] *100 # %로 변환
    
    
### 2020
test_2020=[]


for i in range(0,73):
    test=None    
    test=df[i]['Date'].str.contains('2020')
    test_2020.append(df[i][test])



mdd_2020=[]


historical_max=[]
daily_drawdown=[]
mdd_1=[]


for i in range(0,73):
    historical_max.append(test_2020[i]['Close'].cummax())
    daily_drawdown.append(test_2020[i]['Close'] / historical_max[i] - 1.0)
    mdd_1.append(daily_drawdown[i].cummin())
    mdd_2020.append(mdd_1[i].min())
    mdd_2020[i]= -1 * mdd_2020[i] *100 # %로 변환






stock_name =pd.DataFrame([stock_name[0])


#mdd result
stock_code=pd.DataFrame({'stock_name':stock_name[0]})
mdd_16 =pd.DataFrame({'2016_mdd':mdd_2016})
mdd_17 =pd.DataFrame({'2017_mdd':mdd_2017})
mdd_18 =pd.DataFrame({'2018_mdd':mdd_2018})
mdd_19=pd.DataFrame({'2019_mdd':mdd_2019})
mdd_20=pd.DataFrame({'2020_mdd':mdd_2020})


mdd_result =pd.concat([stock_name[0],mdd_16,mdd_17,mdd_18,mdd_19,mdd_20],axis=1)


