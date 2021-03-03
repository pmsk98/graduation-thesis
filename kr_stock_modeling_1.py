# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:17:53 2021

@author: user
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


for i in df:
    i['diff']=i.Close.diff().shift(-1).fillna(0)
    i['Label'] = None
    
for i in df:
    for e in range(len(i['diff'])):    
        if i['diff'][e] > 0:
            i['Label'][e] = '1'
        elif i['diff'][e]==0:
            i['Label'][e] ='0'
        else:        
            i['Label'][e] = '0'
            


df

for i in df:
    i= i.drop(['diff'],axis=1)




for i in range(0,73):
    df[i]=df[i].drop(['Unnamed: 0','Change'],axis=1)

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
    test=df[i]['Date'].str.contains('2016')
    test_data.append(df[i][test])
    

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
    test=df[i]['Date'].str.contains('2017')
    test_data_2017.append(df[i][test])
    

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
    test=df[i]['Date'].str.contains('2018')
    test_data_2018.append(df[i][test])
    

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
    test=df[i]['Date'].str.contains('2019')
    test_data_2019.append(df[i][test])
    

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
    test=df[i]['Date'].str.contains('2020')
    test_data_2020.append(df[i][test])
    

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
    test=df[i]['Date'].str.contains('2017')
    test_data_2017_2.append(df[i][test])
    

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
    test=df[i]['Date'].str.contains('2018')
    test_data_2018_2.append(df[i][test])
    

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
    test=df[i]['Date'].str.contains('2019')
    test_data_2019_2.append(df[i][test])
    

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
    test=df[i]['Date'].str.contains('2020')
    test_data_2020_2.append(df[i][test])
    

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
    
    
    ###############svm
    svm=SVC()
    
    svm.fit(x_train[i],y_train[i])
    
    pred_svm.append(svm.predict(x_test[i]))
    
    
    #2017
    svm.fit(x_train_1[i],y_train_1[i])
    
    pred_svm_1.append(svm.predict(x_test_1[i]))
    #2018
    svm.fit(x_train_2[i],y_train_2[i])
    
    pred_svm_2.append(svm.predict(x_test_2[i]))
    #2019
    svm.fit(x_train_3[i],y_train_3[i])
    
    pred_svm_3.append(svm.predict(x_test_3[i]))
    #2020
    svm.fit(x_train_4[i],y_train_4[i])
    
    pred_svm_4.append(svm.predict(x_test_4[i]))
    
    #2017_2
    svm.fit(x_train_5[i],y_train_5[i])
    
    pred_svm_5.append(svm.predict(x_test_5[i]))
    #2018_2
    svm.fit(x_train_6[i],y_train_6[i])
    
    pred_svm_6.append(svm.predict(x_test_6[i]))
    #2019_@
    svm.fit(x_train_7[i],y_train_7[i])
    
    pred_svm_7.append(svm.predict(x_test_7[i]))
    #2020_2
    svm.fit(x_train_8[i],y_train_8[i])
    
    pred_svm_8.append(svm.predict(x_test_8[i]))
    
    
    ###############knn
    knn=KNeighborsClassifier()
    
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
    
    voting=VotingClassifier(estimators=[('decison',dt),('knn',knn),('logisitc',logistic),('svm',svm),
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

s




#예측값 붙이기 19년도 데이터에 붙이기

test_2019=[]

for i in range(0,73):    
    test_19=df[i]['Date'].str.contains('2017')
    test_2019.append(df[i][test_19])
    


for i in range(0,73):
    test_2019[i]['pred']=pred_1[i]
    test_2019[i]['pred_decision']=pred_decision_1[i]
    test_2019[i]['pred_naive']=pred_naive_1[i]
    test_2019[i]['pred_randomforest']=pred_randomforest_1[i]
    test_2019[i]['pred_svm']=pred_svm_1[i]
    test_2019[i]['pred_knn']=pred_knn_1[i]
    test_2019[i]['pred_neural']=pred_neural_1[i]
    test_2019[i]['pred_voting']=pred_voting_1[i]
    test_2019[i]['pred_gbm']=pred_gbm_1[i]

#pred 자료형 변경
for i in range(0,73):
    test_2019[i]['pred']=test_2019[i]['pred'].astype('float')
    test_2019[i]['pred_decision']=test_2019[i]['pred_decision'].astype('float')
    test_2019[i]['pred_naive']=test_2019[i]['pred_naive'].astype('float')
    test_2019[i]['pred_randomforest']=test_2019[i]['pred_randomforest'].astype('float')
    test_2019[i]['pred_svm']=test_2019[i]['pred_svm'].astype('float')
    test_2019[i]['pred_knn']=test_2019[i]['pred_knn'].astype('float')
    test_2019[i]['pred_neural']=test_2019[i]['pred_neural'].astype('float')
    test_2019[i]['pred_voting']=test_2019[i]['pred_voting'].astype('float')
    
    

#새로운 라벨 추가(e -> 인덱스 번호)
for i in range(0,73):
    test_2019[i]['position']=None
    
                       
#randomforest
for i in range(0,73):
    for e in test_2019[i].index:
        try:
            if test_2019[i]['pred'][e]+test_2019[i]['pred'][e+1]==0:
                test_2019[i]['position'][e+1]='no action'
            elif test_2019[i]['pred'][e]+test_2019[i]['pred'][e+1]==2:
                test_2019[i]['position'][e+1]='holding'
            elif test_2019[i]['pred'][e] > test_2019[i]['pred'][e+1]:
                test_2019[i]['position'][e+1]='sell'
            else:
                test_2019[i]['position'][e+1]='buy'
        except:
            pass

#첫날 position이 holding일 경우 buy로 변경
for i in range(0,73):
    if test_2019[i]['position'][test_2019[i].index[1]]=='holding':
        test_2019[i]['position'][test_2019[i].index[1]]='buy'
    elif test_2019[i]['position'][test_2019[i].index[1]]=='sell':
        test_2019[i]['position'][test_2019[i].index[1]]='buy'
    else:
        print(i)


#강제 청산
for i in range(0,73):
    for e in test_2019[i].index[-1:]:
        if test_2019[i]['position'][e]=='holding':
            test_2019[i]['position'][e]='sell'
        elif test_2019[i]['position'][e]=='buy':
            test_2019[i]['position'][e]='sell'
        elif test_2019[i]['position'][e]=='no action':
            test_2019[i]['position'][e]='sell'
        else:
            print(i)

pred_gbm_1

for i in range(0,73):
    test_2019[i]['profit']=None
    
#다음날 수정 종가를 가져오게 생성
for i in range(0,73):
    for e in test_2019[i].index:
        try:
            if test_2019[i]['position'][e]=='buy':
                test_2019[i]['profit'][e]=test_2019[i]['Close'][e+1]
            elif test_2019[i]['position'][e]=='sell':
                test_2019[i]['profit'][e]=test_2019[i]['Close'][e+1]
            else:
                print(i)
        except:
            pass



for i in range(0,73):
    for e in test_2019[i].index[-1:]:
        if test_2019[i]['position'][e]=='sell':
            test_2019[i]['profit'][e]=test_2019[i]['Close'][e]
        
####

buy_label=[]
for i in range(0,73):
    buy_position=test_2019[i]['position']=='buy'
    buy_label.append(test_2019[i][buy_position])
    
sell_label=[]
for i in range(0,73):
    sell_position=test_2019[i]['position']=='sell'
    sell_label.append(test_2019[i][sell_position])    


buy=[]
sell=[]
for i in range(0,73):
    buy.append(buy_label[i]['Close'].reset_index(drop=True))
    sell.append(sell_label[i]['Close'].reset_index(drop=True))
    
  
profit_2=[]    
for i in range(0,73):
    profit_2.append(sell[i]-buy[i])
  

for i in range(0,73):
    test_2019[i]['profit_2']=None
    

#profit 결측치 처리
for i in range(0,73):
    profit_2[i]=profit_2[i].dropna()
    
    
#profit_2 sell에 해당하는 행에 값 넣기
for tb, pf in zip(test_2019, profit_2):
    total_idx = tb[tb['position'] == 'sell'].index
    total_pf_idx = pf.index
    for idx, pf_idx in zip(total_idx, total_pf_idx):
        tb.loc[idx, 'profit_2'] = pf[pf_idx]
        

for i in range(0,73):
    test_2019[i]['profit_cumsum']=None
    
    
    

#profit 누적 합 
for i in range(0,73):
    for e in test_2019[i].index:
        try:
            if test_2019[i]['position'][e]=='holding':
                test_2019[i]['profit_2'][e]=0
            elif test_2019[i]['position'][e]=='no action':
                test_2019[i]['profit_2'][e]=0
            elif test_2019[i]['position'][e]=='buy':
                test_2019[i]['profit_2'][e]=0
            else:
                print(i)
        except:
            pass


#새로운 청산 기준 누적합

for i in range(0,73):
    test_2019[i]['profit_cumsum2']=None    
    
    
for i in range(0,73):
    test_2019[i]['profit_cumsum']=test_2019[i]['profit_2'].cumsum()


#############################ratio 작성

#ratio 작성
for i in range(0,73):
    profit_2[i]=pd.DataFrame(profit_2[i])

#거래횟수
trade= []

for i in range(0,73):
    trade.append(len(profit_2[i]))
    
#승률


for i in range(0,73):
    profit_2[i]['average']=None

   
for i in range(0,73):
    for e in range(len(profit_2[i])):      
        if profit_2[i]['Close'][e] > 0:
            profit_2[i]['average'][e]='gain'
        else:
            profit_2[i]['average'][e]='loss'
            
for i in range(0,73):
    for e in range(len(profit_2[i])):
        if profit_2[i]['Close'][e] < 0:
            profit_2[i]['Close'][e]=profit_2[i]['Close'][e] * -1
        else:
            print(i)

win=[]
for i in range(0,73):
    try:
        win.append(profit_2[i].groupby('average').size()[0]/len(profit_2[i]))
    except:
        win.append('0')
    
#평균 수익

gain=[]

for i in range(0,73):
    gain.append(profit_2[i].groupby('average').mean())
    

real_gain=[]

for i in range(0,73):
    try:
        real_gain.append(gain[i]['Close'][0])
    except:
        real_gain.append('0')



#평균 손실
loss=[]

for i in range(0,73):
    try:
        loss.append(gain[i]['Close'][1])
    except:
        loss.append('0')

    
loss
#payoff ratio
payoff=[]

for i in range(0,73):
    try:
        payoff.append(gain[i]['Close'][0]/gain[i]['Close'][1])
    except:
        payoff.append('inf')
    
#profit factor

factor_sum=[]

len(factor_sum)
for i in range(0,73):
    factor_sum.append(profit_2[i].groupby('average').sum())

factor=[]

for i in range(0,73):
    try:
        factor.append(factor_sum[i]['Close'][0]/factor_sum[i]['Close'][1])
    except:
        factor.append('0')

#year
year=[]

for i in range(0,73):
    year.append('2017')

#최종 결과물 파일 작성
stock_name=pd.DataFrame({'stock_name':file_list})

stock_name=stock_name.replace('.csv','',regex=True)

year=pd.DataFrame({'year':year})

trade=pd.DataFrame({'No.trades':trade})

win=pd.DataFrame({'Win%':win})

real_gain=pd.DataFrame({'Average gain($)':real_gain})

loss=pd.DataFrame({'Average loss($)':loss})

payoff=pd.DataFrame({'Payoff ratio':payoff})

factor=pd.DataFrame({'Profit factor':factor})


#########################
#2016
result =pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)
#2017
result1=pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)
#2018
result2=pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)
#2019
result3=pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)
#2020
result4=pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)


final_result=pd.concat([result,result1,result2,result3,result4])


final_result.to_csv('kr_stock_ratio_method1_randomforest.csv',encoding='euc-kr')

######### 두 번째 방법 ################# 여기서부터
#2016
result

#2017
result5=pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)

#2018
result6=pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)

#2019
result7=pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)

#2020
result8=pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)


final_result2=pd.concat([result,result5,result6,result7,result8])


final_result2.to_csv('kr_stock_ratio_method2_voting.csv',encoding='euc-kr')


        
