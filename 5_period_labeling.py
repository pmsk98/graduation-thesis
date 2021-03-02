# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:26:42 2021

@author: user
"""

import glob
import os
import pandas as pd
import  talib
import numpy as np
import math

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


for  i in df:
    ADX=talib.ADX(i.High,i.Low,i.Close,timeperiod=14)
 
    aroondown,aroonup =talib.AROON(i.High, i.Low, timeperiod=14)
    
    AROONOSC=talib.AROONOSC(i.High,i.Low,timeperiod=14)
    
    BOP=talib.BOP(i.Open,i.High,i.Low,i.Close)
    
    CCI=talib.CCI(i.High,i.Low,i.Close,timeperiod=9)
    
    CMO=talib.CMO(i.Close,timeperiod=14)
    
    DX=talib.DX(i.High,i.Low,i.Close,timeperiod=14)
    
    MFI=talib.MFI(i.High, i.Low,i.Close, i.Volume, timeperiod=14)
    
    PPO=talib.PPO(i.Close, fastperiod=12, slowperiod=26, matype=0)
    
    ROC=talib.ROC(i.Close,timeperiod=10)
    
    RSI=talib.RSI(i.Close,timeperiod=14)
    
    slowk, slowd = talib.STOCH(i.High, i.Low, i.Close, fastk_period=12.5, slowk_period=5, slowk_matype=0, slowd_period=3, slowd_matype=0)
    
    fastk, fastd = talib.STOCHF(i.High, i.Low, i.Close, fastk_period=5, fastd_period=5.3, fastd_matype=0)
    
    ULTOSC=talib.ULTOSC(i.High,i.Low,i.Close,timeperiod1=7,timeperiod2=14,timeperiod3=28)
    
    WILLR=talib.WILLR(i.High,i.Low,i.Close,timeperiod=14)
        
    i['ADX']=ADX
    i['aroondown']=aroondown
    i['aroonup']=aroonup
    i['BOP']=BOP
    i['CCI']=CCI
    i['CMO']=CMO
    i['DX']=DX
    i['MFI']=MFI
    i['PPO']=PPO
    i['ROC']=ROC
    i['RSI']=RSI
    i['slowk']=slowk
    i['slowd']=slowd
    i['fastk']=fastk
    i['fastd']=fastd
    i['ULTOSC']=ULTOSC
    i['WILLR']=WILLR


for i in range(len(df)):
    temp = df[i]
    deleted = []
    for year in range(2008, 2021):
        deleted.append(temp[('%d-01-01' %year <= temp['Date']) & (temp['Date'] < '%d-01-01' %(year+1))].iloc[:-4])
    df[i] = pd.concat(deleted)



for i in df:
    i['diff']=i.Close.diff().shift(-1).fillna(0)
    i['Label'] = None
    i['new_return']=None
    i['return'] = None
    

for i in df:
    i=i.drop(['Unnamed: 0'],axis=1)



#수익률 표준편차 계산

for i in df:
    i['return']=i['Close'].pct_change()*100


for i in range(0,73):
    for e in df[i].index:
        try:
            df[i]['new_return'][e] = df[i]['return'][e+5] - df[i]['return'][e+1]
        except:
            pass




#train set std_return
std_return =[]

len(std_return)


for i in df:    
    std_return.append(np.std(i['new_return']))


#up/down 기준 생성 
up_label =[]
down_label= []

for i in range(0,73):
    up_label.append(0.25*std_return[i])
    down_label.append(-0.25 * std_return[i])
    
    

#label 생성
for i in range(0,73):
    for e in df[i].index:
        if df[i]['return'][e] > up_label[i]:
            df[i]['Label'][e]= '1'
        elif df[i]['return'][e] < down_label[i]:
            df[i]['Label'][e]='0'
        else:
            None


#결측치와 필요없는 변수 삭제
for i in range(0,73):
    df[i]=df[i].drop(['Unnamed: 0','return','Change','new_return'],axis=1)

#결측치 처리
for i in range(0,73):
    df[i]=df[i].dropna(axis=0)

#인덱스 번호 한번더 초기화
for i in range(0,73):
    df[i]=df[i].reset_index()
    df[i]=df[i].drop(['index'],axis=1)


#model train/test set 생성  
train_data=[]
test_data=[]

############2016
for i in range(0,73):
    train=None
    train=df[i]['Date'].str.contains('2009|2010|2011|2012|2013|2014|2015')
    train_data.append(df[i][train])
for i in range(0,73):
    test=None    
    test=df[i]['Date'].str.contains('2016')
    test_data.append(df[i][test])
    



for i in range(0,73):
    train_data[i]=train_data[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)
    test_data[i]=test_data[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)




import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

font_path = r'C:/Users/user/Desktop/연구/NanumBarunGothic.ttf'
fontprop = fm.FontProperties(fname=font_path, size=18)


for i in range(0,5):
    train_data[i]['Label'].value_counts().plot(kind='bar')
    plt.title('{}_train data label'.format(file_list[i]),fontproperties=fontprop)
    plt.show()


label_count=[]
for i in range(0,5):
   label_count.append(train_data[i]['Label'].value_counts())