# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 12:53:58 2021

@author: user
"""

files=glob.glob('C:/Users/user/Desktop/연구/시가총액100위/*.csv')

file_list =os.listdir(path)

path = "C:/Users/user/Desktop/연구/시가총액100위"


file_list =os.listdir(path)
file_list


kr_stock=[]


for file in file_list:
    path = "C:/Users/user/Desktop/연구/시가총액100위"
    data=pd.read_csv(path+"/"+file)
    if data['Date'][0] <= '2009-01-01':
        kr_stock.append(data)
        data.to_csv('{}'.format(file))
    else:
        pass
    
    


#주가 동향 확인

from datetime import datetime
import time
import matplotlib.font_manager as fm


path='C:/Users/user/Desktop/연구/NanumBarunGothic.ttf'
fontprop =fm.FontProperties(fname=path,size=15)



time_format="%Y-%m-%d"



for i in range(0,75):
    df[i]['Date']=pd.to_datetime(df[i]['Date'],format=time_format)
    


for i in range(0,75):
    
    plt.title('{}'.format(file_list[i]),fontproperties=fontprop)
    plt.plot(df[i][['Date']],df[i][['Close']])
    plt.show()

