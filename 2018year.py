# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 18:31:16 2021

@author: 박명석
"""

test_2019=[]

for i in range(0,73):    
    test_19=df[i]['Date'].str.contains('2018')
    test_2019.append(df[i][test_19])
    


for i in range(0,73):
    test_2019[i]['pred']=pred_2[i]
    test_2019[i]['pred_decision']=pred_decision_2[i]
    test_2019[i]['pred_naive']=pred_naive_2[i]
    test_2019[i]['pred_randomforest']=pred_randomforest_2[i]
    test_2019[i]['pred_svm']=pred_svm_2[i]
    test_2019[i]['pred_knn']=pred_knn_2[i]
    test_2019[i]['pred_neural']=pred_neural_2[i]
    test_2019[i]['pred_voting']=pred_voting_2[i]
    test_2019[i]['pred_gbm']=pred_gbm_2[i]
    

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
    test_2019[i]['pred_gbm']=test_2019[i]['pred_gbm'].astype('float')
    
    

#새로운 라벨 추가(e -> 인덱스 번호)
for i in range(0,73):
    test_2019[i]['position']=None
    
                       
#randomforest
for i in range(0,73):
    for e in test_2019[i].index:
        try:
            if test_2019[i]['pred_gbm'][e]+test_2019[i]['pred_gbm'][e+1]==0:
                test_2019[i]['position'][e+1]='no action'
            elif test_2019[i]['pred_gbm'][e]+test_2019[i]['pred_gbm'][e+1]==2:
                test_2019[i]['position'][e+1]='holding'
            elif test_2019[i]['pred_gbm'][e] > test_2019[i]['pred_gbm'][e+1]:
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
    year.append('2018')

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

#2018
result2 =pd.concat([year,stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)