# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:17:05 2021

@author: pmsk9
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:03:48 2021

@author: pmsk9
"""

import glob
import os
import pandas as pd
import  talib
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

path = "C:/Users/pmsk9/Desktop/2021년도 연구/시가총액100위"

file_list =os.listdir(path)

df= []

for file in file_list:
    path = "C:/Users/pmsk9/Desktop/2021년도 연구/시가총액100위"
    
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

##
test_df=df.copy()

for i in test_df:
    i['diff']=i.Close.diff().shift(-1).fillna(0)
    i['Label'] = None
    i['new_return']=None
    i['return'] = None


for i in range(0,73):
    test_df[i]=test_df[i].drop(['Unnamed: 0','Change','new_return','return'],axis=1)

##


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



for i in range(len(df)):
    temp = df[i]
    deleted = []
    for year in range(2008, 2021):
        deleted.append(temp[('%d-01-01' %year <= temp['Date']) & (temp['Date'] < '%d-01-01' %(year+1))].iloc[:-9])
    df[i] = pd.concat(deleted)


for i in range(0,73):
    for e in df[i].index:
        try:
            df[i]['new_return'][e] = df[i]['return'][e+10] - df[i]['return'][e+1]
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
    up_label.append(1*std_return[i])
    down_label.append(-1 * std_return[i])
    
    

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


###########modeling

#model train/test set 생성  
train_data=[]
test_data=[]


train_data_2017=[]
test_data_2017=[]

train_data_2018=[]
test_data_2018=[]

train_data_2019=[]
test_data_2019=[]


train_data_2020=[]
test_data_2020=[]

train_data_2017_2=[]
test_data_2017_2=[]

train_data_2018_2=[]
test_data_2018_2=[]

train_data_2019_2=[]
test_data_2019_2=[]

train_data_2020_2=[]
test_data_2020_2=[]
############2016
for i in range(0,73):
    train=None
    train=df[i]['Date'].str.contains('2009|2010|2011|2012|2013|2014|2015')
    train_data.append(df[i][train])
for i in range(0,73):
    test=None    
    test=test_df[i]['Date'].str.contains('2016')
    test_data.append(test_df[i][test])
    

for i in range(0,73):
    train_data[i]=train_data[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)
    test_data[i]=test_data[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)
    

###############2017_1
for i in range(0,73):
    train=None
    train=df[i]['Date'].str.contains('2009|2010|2011|2012|2013|2014|2015|2016')
    train_data_2017.append(df[i][train])
for i in range(0,73):
    test=None      
    test=test_df[i]['Date'].str.contains('2017')
    test_data_2017.append(test_df[i][test])
    

for i in range(0,73):
    train_data_2017[i]=train_data_2017[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)
    test_data_2017[i]=test_data_2017[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

############2018_1
for i in range(0,73):
    train=None
    train=df[i]['Date'].str.contains('2009|2010|2011|2012|2013|2014|2015|2016|2017')
    train_data_2018.append(df[i][train])
for i in range(0,73):   
    test=None
    test=test_df[i]['Date'].str.contains('2018')
    test_data_2018.append(test_df[i][test])
    

for i in range(0,73):
    train_data_2018[i]=train_data_2018[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)
    test_data_2018[i]=test_data_2018[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)
################2019_1
for i in range(0,73):
    train=None
    train=df[i]['Date'].str.contains('2009|2010|2011|2012|2013|2014|2015|2016|2017|2018')
    train_data_2019.append(df[i][train])
for i in range(0,73):  
    test=None
    test=test_df[i]['Date'].str.contains('2019')
    test_data_2019.append(test_df[i][test])
    

for i in range(0,73):
    train_data_2019[i]=train_data_2019[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)
    test_data_2019[i]=test_data_2019[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)
################2020_1
for i in range(0,73):
    train=None
    train=df[i]['Date'].str.contains('2009|2010|2011|2012|2013|2014|2015|2016|2017|2018|2019')
    train_data_2020.append(df[i][train])
for i in range(0,73):
    test=None    
    test=test_df[i]['Date'].str.contains('2020')
    test_data_2020.append(test_df[i][test])
    

for i in range(0,73):
    train_data_2020[i]=train_data_2020[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)
    test_data_2020[i]=test_data_2020[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)


#############2017_2
for i in range(0,73):
    train=None
    train=df[i]['Date'].str.contains('2010|2011|2012|2013|2014|2015|2016')
    train_data_2017_2.append(df[i][train])
for i in range(0,73):
    test=None    
    test=test_df[i]['Date'].str.contains('2017')
    test_data_2017_2.append(test_df[i][test])
    

for i in range(0,73):
    train_data_2017_2[i]=train_data_2017_2[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)
    test_data_2017_2[i]=test_data_2017_2[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)
############2018_2
for i in range(0,73):
    train=None
    train=df[i]['Date'].str.contains('2011|2012|2013|2014|2015|2016|2017')
    train_data_2018_2.append(df[i][train])
for i in range(0,73):
    test=None    
    test=test_df[i]['Date'].str.contains('2018')
    test_data_2018_2.append(test_df[i][test])
    

for i in range(0,73):
    train_data_2018_2[i]=train_data_2018_2[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)
    test_data_2018_2[i]=test_data_2018_2[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

#############2019_2
for i in range(0,73):
    train=None
    train=df[i]['Date'].str.contains('2012|2013|2014|2015|2016|2017|2018')
    train_data_2019_2.append(df[i][train])
for i in range(0,73):
    test=None    
    test=test_df[i]['Date'].str.contains('2019')
    test_data_2019_2.append(test_df[i][test])
    

for i in range(0,73):
    train_data_2019_2[i]=train_data_2019_2[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)
    test_data_2019_2[i]=test_data_2019_2[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

###############2020_2
for i in range(0,73):
    train=None
    train=df[i]['Date'].str.contains('2013|2014|2015|2016|2017|2018|2019')
    train_data_2020_2.append(df[i][train])
for i in range(0,73):  
    test=None
    test=test_df[i]['Date'].str.contains('2020')
    test_data_2020_2.append(test_df[i][test])
    

for i in range(0,73):
    train_data_2020_2[i]=train_data_2020_2[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)
    test_data_2020_2[i]=test_data_2020_2[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)





#x_train,y_train,x_test,y_test

x_train =[]
y_train =[]
x_test=[]
y_test=[]

x_train_1=[]
y_train_1=[]
x_test_1=[]
y_test_1=[]


x_train_2=[]
y_train_2=[]
x_test_2=[]
y_test_2=[]


x_train_3=[]
y_train_3=[]
x_test_3=[]
y_test_3=[]

x_train_4=[]
y_train_4=[]
x_test_4=[]
y_test_4=[]

x_train_5=[]
y_train_5=[]
x_test_5=[]
y_test_5=[]


x_train_6=[]
y_train_6=[]
x_test_6=[]
y_test_6=[]


x_train_7=[]
y_train_7=[]
x_test_7=[]
y_test_7=[]


x_train_8=[]
y_train_8=[]
x_test_8=[]
y_test_8=[]

#######2016
for i in range(0,73):
    x_train.append(train_data[i].drop(['Label'],axis=1))
    y_train.append(train_data[i]['Label'])
    
    x_test.append(test_data[i].drop(['Label'],axis=1))
    y_test.append(test_data[i]['Label']) 
########2017_1
for i in range(0,73):
    x_train_1.append(train_data_2017[i].drop(['Label'],axis=1))
    y_train_1.append(train_data_2017[i]['Label'])
    
    x_test_1.append(test_data_2017[i].drop(['Label'],axis=1))
    y_test_1.append(test_data_2017[i]['Label']) 

#########2018_1
for i in range(0,73):
    x_train_2.append(train_data_2018[i].drop(['Label'],axis=1))
    y_train_2.append(train_data_2018[i]['Label'])
    
    x_test_2.append(test_data_2018[i].drop(['Label'],axis=1))
    y_test_2.append(test_data_2018[i]['Label']) 

##########2019_1
for i in range(0,73):
    x_train_3.append(train_data_2019[i].drop(['Label'],axis=1))
    y_train_3.append(train_data_2019[i]['Label'])
    
    x_test_3.append(test_data_2019[i].drop(['Label'],axis=1))
    y_test_3.append(test_data_2019[i]['Label']) 

#############2020_1
for i in range(0,73):
    x_train_4.append(train_data_2020[i].drop(['Label'],axis=1))
    y_train_4.append(train_data_2020[i]['Label'])
    
    x_test_4.append(test_data_2020[i].drop(['Label'],axis=1))
    y_test_4.append(test_data_2020[i]['Label']) 
    
##############2017_@
for i in range(0,73):
    x_train_5.append(train_data_2017_2[i].drop(['Label'],axis=1))
    y_train_5.append(train_data_2017_2[i]['Label'])
    
    x_test_5.append(test_data_2017_2[i].drop(['Label'],axis=1))
    y_test_5.append(test_data_2017_2[i]['Label'])


############2018_2
for i in range(0,73):
    x_train_6.append(train_data_2018_2[i].drop(['Label'],axis=1))
    y_train_6.append(train_data_2018_2[i]['Label'])
    
    x_test_6.append(test_data_2018_2[i].drop(['Label'],axis=1))
    y_test_6.append(test_data_2018_2[i]['Label']) 

############2019_2
for i in range(0,73):
    x_train_7.append(train_data_2019_2[i].drop(['Label'],axis=1))
    y_train_7.append(train_data_2019_2[i]['Label'])
    
    x_test_7.append(test_data_2019_2[i].drop(['Label'],axis=1))
    y_test_7.append(test_data_2019_2[i]['Label']) 

############2020_2
for i in range(0,73):
    x_train_8.append(train_data_2020_2[i].drop(['Label'],axis=1))
    y_train_8.append(train_data_2020_2[i]['Label'])
    
    x_test_8.append(test_data_2020_2[i].drop(['Label'],axis=1))
    y_test_8.append(test_data_2020_2[i]['Label']) 


    
#모델링
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier




pred=[]
pred_decision=[]
pred_naive=[]
pred_randomforest=[]
pred_svm=[]
pred_knn=[]
pred_neural=[]
pred_voting=[]
pred_gbm=[]

pred_1=[]
pred_decision_1=[]
pred_naive_1=[]
pred_randomforest_1=[]
pred_svm_1=[]
pred_knn_1=[]
pred_neural_1=[]
pred_voting_1=[]
pred_gbm_1=[]


pred_2=[]
pred_decision_2=[]
pred_naive_2=[]
pred_randomforest_2=[]
pred_svm_2=[]
pred_knn_2=[]
pred_neural_2=[]
pred_voting_2=[]
pred_gbm_2=[]

pred_3=[]
pred_decision_3=[]
pred_naive_3=[]
pred_randomforest_3=[]
pred_svm_3=[]
pred_knn_3=[]
pred_neural_3=[]
pred_voting_3=[]
pred_gbm_3=[]

pred_4=[]
pred_decision_4=[]
pred_naive_4=[]
pred_randomforest_4=[]
pred_svm_4=[]
pred_knn_4=[]
pred_neural_4=[]
pred_voting_4=[]
pred_gbm_4=[]

pred_5=[]
pred_decision_5=[]
pred_naive_5=[]
pred_randomforest_5=[]
pred_svm_5=[]
pred_knn_5=[]
pred_neural_5=[]
pred_voting_5=[]
pred_gbm_5=[]


pred_6=[]
pred_decision_6=[]
pred_naive_6=[]
pred_randomforest_6=[]
pred_svm_6=[]
pred_knn_6=[]
pred_neural_6=[]
pred_voting_6=[]
pred_gbm_6=[]


pred_7=[]
pred_decision_7=[]
pred_naive_7=[]
pred_randomforest_7=[]
pred_svm_7=[]
pred_knn_7=[]
pred_neural_7=[]
pred_voting_7=[]
pred_gbm_7=[]

pred_8=[]
pred_decision_8=[]
pred_naive_8=[]
pred_randomforest_8=[]
pred_svm_8=[]
pred_knn_8=[]
pred_neural_8=[]
pred_voting_8=[]
pred_gbm_8=[]

for i in range(0,73):
    #logistic
    logistic =LogisticRegression()
    logistic.fit(x_train[i],y_train[i])
    
    pred.append(logistic.predict(x_test[i]))
    #2017
    logistic.fit(x_train_1[i],y_train_1[i])
    
    pred_1.append(logistic.predict(x_test_1[i]))
    #2018
    logistic.fit(x_train_2[i],y_train_2[i])
    
    pred_2.append(logistic.predict(x_test_2[i]))
    #2019
    logistic.fit(x_train_3[i],y_train_3[i])
    
    pred_3.append(logistic.predict(x_test_3[i]))
    #2020
    logistic.fit(x_train_4[i],y_train_4[i])
    
    pred_4.append(logistic.predict(x_test_4[i]))
    
    #2017_2
    logistic.fit(x_train_5[i],y_train_5[i])
    
    pred_5.append(logistic.predict(x_test_5[i]))
    #2018_2
    logistic.fit(x_train_6[i],y_train_6[i])
    
    pred_6.append(logistic.predict(x_test_6[i]))
    #2019_@
    logistic.fit(x_train_7[i],y_train_7[i])
    
    pred_7.append(logistic.predict(x_test_7[i]))
    #2020_2
    logistic.fit(x_train_8[i],y_train_8[i])
    
    pred_8.append(logistic.predict(x_test_8[i]))
    
    
    ##############decision tree
    dt=DecisionTreeClassifier()
    
    dt.fit(x_train[i],y_train[i])
    pred_decision.append(dt.predict(x_test[i]))
    
    #2017
    dt.fit(x_train_1[i],y_train_1[i])
    
    pred_decision_1.append(dt.predict(x_test_1[i]))
    #2018
    dt.fit(x_train_2[i],y_train_2[i])
    
    pred_decision_2.append(dt.predict(x_test_2[i]))
    #2019
    dt.fit(x_train_3[i],y_train_3[i])
    
    pred_decision_3.append(dt.predict(x_test_3[i]))
    #2020
    dt.fit(x_train_4[i],y_train_4[i])
    
    pred_decision_4.append(dt.predict(x_test_4[i]))
    
    #2017_2
    dt.fit(x_train_5[i],y_train_5[i])
    
    pred_decision_5.append(dt.predict(x_test_5[i]))
    #2018_2
    dt.fit(x_train_6[i],y_train_6[i])
    
    pred_decision_6.append(dt.predict(x_test_6[i]))
    #2019_@
    dt.fit(x_train_7[i],y_train_7[i])
    
    pred_decision_7.append(dt.predict(x_test_7[i]))
    #2020_2
    dt.fit(x_train_8[i],y_train_8[i])
    
    pred_decision_8.append(dt.predict(x_test_8[i]))
    
    
    ##############naive
    naive=GaussianNB()
    
    naive.fit(x_train[i],y_train[i])
    
    pred_naive.append(naive.predict(x_test[i]))
    
    #2017
    naive.fit(x_train_1[i],y_train_1[i])
    
    pred_naive_1.append(naive.predict(x_test_1[i]))
    #2018
    naive.fit(x_train_2[i],y_train_2[i])
    
    pred_naive_2.append(naive.predict(x_test_2[i]))
    #2019
    naive.fit(x_train_3[i],y_train_3[i])
    
    pred_naive_3.append(naive.predict(x_test_3[i]))
    #2020
    naive.fit(x_train_4[i],y_train_4[i])
    
    pred_naive_4.append(naive.predict(x_test_4[i]))
    
    #2017_2
    naive.fit(x_train_5[i],y_train_5[i])
    
    pred_naive_5.append(naive.predict(x_test_5[i]))
    #2018_2
    naive.fit(x_train_6[i],y_train_6[i])
    
    pred_naive_6.append(naive.predict(x_test_6[i]))
    #2019_@
    naive.fit(x_train_7[i],y_train_7[i])
    
    pred_naive_7.append(naive.predict(x_test_7[i]))
    #2020_2
    naive.fit(x_train_8[i],y_train_8[i])
    
    pred_naive_8.append(naive.predict(x_test_8[i]))
    
    
    #############randomforest
    randomforest=RandomForestClassifier()
    
    randomforest.fit(x_train[i],y_train[i])
    
    pred_randomforest.append(randomforest.predict(x_test[i]))
    
    #2017
    randomforest.fit(x_train_1[i],y_train_1[i])
    
    pred_randomforest_1.append(randomforest.predict(x_test_1[i]))
    #2018
    randomforest.fit(x_train_2[i],y_train_2[i])
    
    pred_randomforest_2.append(randomforest.predict(x_test_2[i]))
    #2019
    randomforest.fit(x_train_3[i],y_train_3[i])
    
    pred_randomforest_3.append(randomforest.predict(x_test_3[i]))
    #2020
    randomforest.fit(x_train_4[i],y_train_4[i])
    
    pred_randomforest_4.append(randomforest.predict(x_test_4[i]))
    
    #2017_2
    randomforest.fit(x_train_5[i],y_train_5[i])
    
    pred_randomforest_5.append(randomforest.predict(x_test_5[i]))
    #2018_2
    randomforest.fit(x_train_6[i],y_train_6[i])
    
    pred_randomforest_6.append(randomforest.predict(x_test_6[i]))
    #2019_@
    randomforest.fit(x_train_7[i],y_train_7[i])
    
    pred_randomforest_7.append(randomforest.predict(x_test_7[i]))
    #2020_2
    randomforest.fit(x_train_8[i],y_train_8[i])
    
    pred_randomforest_8.append(randomforest.predict(x_test_8[i]))
    

    
    
    ###############knn
    knn=KNeighborsClassifier(n_neighbors=3)
    
    knn.fit(x_train[i],y_train[i])
    
    pred_knn.append(knn.predict(x_test[i]))
    
    
    #2017
    knn.fit(x_train_1[i],y_train_1[i])
    
    pred_knn_1.append(knn.predict(x_test_1[i]))
    #2018
    knn.fit(x_train_2[i],y_train_2[i])
    
    pred_knn_2.append(knn.predict(x_test_2[i]))
    #2019
    knn.fit(x_train_3[i],y_train_3[i])
    
    pred_knn_3.append(knn.predict(x_test_3[i]))
    #2020
    knn.fit(x_train_4[i],y_train_4[i])
    
    pred_knn_4.append(knn.predict(x_test_4[i]))
    
    #2017_2
    knn.fit(x_train_5[i],y_train_5[i])
    
    pred_knn_5.append(knn.predict(x_test_5[i]))
    #2018_2
    knn.fit(x_train_6[i],y_train_6[i])
    
    pred_knn_6.append(knn.predict(x_test_6[i]))
    #2019_@
    knn.fit(x_train_7[i],y_train_7[i])
    
    pred_knn_7.append(knn.predict(x_test_7[i]))
    #2020_2
    knn.fit(x_train_8[i],y_train_8[i])
    
    pred_knn_8.append(knn.predict(x_test_8[i]))
    
    ###############nueral
    
    nueral=MLPClassifier()
    
    nueral.fit(x_train[i],y_train[i])
    
    pred_neural.append(nueral.predict(x_test[i]))
    
    
    #2017
    nueral.fit(x_train_1[i],y_train_1[i])
    
    pred_neural_1.append(nueral.predict(x_test_1[i]))
    #2018
    nueral.fit(x_train_2[i],y_train_2[i])
    
    pred_neural_2.append(nueral.predict(x_test_2[i]))
    #2019
    nueral.fit(x_train_3[i],y_train_3[i])
    
    pred_neural_3.append(nueral.predict(x_test_3[i]))
    #2020
    nueral.fit(x_train_4[i],y_train_4[i])
    
    pred_neural_4.append(nueral.predict(x_test_4[i]))
    
    #2017_2
    nueral.fit(x_train_5[i],y_train_5[i])
    
    pred_neural_5.append(nueral.predict(x_test_5[i]))
    #2018_2
    nueral.fit(x_train_6[i],y_train_6[i])
    
    pred_neural_6.append(nueral.predict(x_test_6[i]))
    #2019_@
    nueral.fit(x_train_7[i],y_train_7[i])
    
    pred_neural_7.append(nueral.predict(x_test_7[i]))
    #2020_2
    nueral.fit(x_train_8[i],y_train_8[i])
    
    pred_neural_8.append(nueral.predict(x_test_8[i]))
    
    
    
    ###########voting
    
    voting=VotingClassifier(estimators=[('decison',dt),('knn',knn),('logisitc',logistic),
                                        ('naive',naive),('nueral',nueral)],voting='hard')
    
    voting.fit(x_train[i],y_train[i])
    
    pred_voting.append(voting.predict(x_test[i]))
    
    
    #2017
    voting.fit(x_train_1[i],y_train_1[i])
    
    pred_voting_1.append(voting.predict(x_test_1[i]))
    #2018
    voting.fit(x_train_2[i],y_train_2[i])
    
    pred_voting_2.append(voting.predict(x_test_2[i]))
    #2019
    voting.fit(x_train_3[i],y_train_3[i])
    
    pred_voting_3.append(voting.predict(x_test_3[i]))
    #2020
    voting.fit(x_train_4[i],y_train_4[i])
    
    pred_voting_4.append(voting.predict(x_test_4[i]))
    
    #2017_2
    voting.fit(x_train_5[i],y_train_5[i])
    
    pred_voting_5.append(voting.predict(x_test_5[i]))
    #2018_2
    voting.fit(x_train_6[i],y_train_6[i])
    
    pred_voting_6.append(voting.predict(x_test_6[i]))
    #2019_@
    voting.fit(x_train_7[i],y_train_7[i])
    
    pred_voting_7.append(voting.predict(x_test_7[i]))
    #2020_2
    voting.fit(x_train_8[i],y_train_8[i])
    
    pred_voting_8.append(voting.predict(x_test_8[i]))
    
    ########gbm
    gbm=GradientBoostingClassifier(random_state=0)
    
    gbm.fit(x_train[i],y_train[i])
    
    pred_gbm.append(gbm.predict(x_test[i]))
    
    
    #2017
    gbm.fit(x_train_1[i],y_train_1[i])
    
    pred_gbm_1.append(gbm.predict(x_test_1[i]))
    #2018
    gbm.fit(x_train_2[i],y_train_2[i])
    
    pred_gbm_2.append(gbm.predict(x_test_2[i]))
    #2019
    gbm.fit(x_train_3[i],y_train_3[i])
    
    pred_gbm_3.append(gbm.predict(x_test_3[i]))
    #2020
    gbm.fit(x_train_4[i],y_train_4[i])
    
    pred_gbm_4.append(gbm.predict(x_test_4[i]))
    
    #2017_2
    gbm.fit(x_train_5[i],y_train_5[i])
    
    pred_gbm_5.append(gbm.predict(x_test_5[i]))
    #2018_2
    gbm.fit(x_train_6[i],y_train_6[i])
    
    pred_gbm_6.append(gbm.predict(x_test_6[i]))
    #2019_@
    gbm.fit(x_train_7[i],y_train_7[i])
    
    pred_gbm_7.append(gbm.predict(x_test_7[i]))
    #2020_2
    gbm.fit(x_train_8[i],y_train_8[i])
    
    pred_gbm_8.append(gbm.predict(x_test_8[i]))