# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 20:49:12 2018

@author: liuxy
"""

import pandas as pd
import numpy as np
import quandl
# find data of S&P500 and Russell 2000
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import bs4 as bs
import pickle
import requests
from datetime import datetime
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
def save_sp500_tickers():
   resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
   soup = bs.BeautifulSoup(resp.text, 'lxml')
   table = soup.find('table', {'class': 'wikitable sortable'})
   tickers = []
   for row in table.findAll('tr')[1:]:
       ticker = row.findAll('td')[0].text
       tickers.append(ticker)

   with open("sp500tickers.pickle","wb") as f:
       pickle.dump(tickers,f)

   return tickers

sp500_list=save_sp500_tickers()

error_list=list()
megadata={}

for i in range(len(sp500_list)):
   try:
       megadata[sp500_list[i]]=quandl.get("WIKI/"+sp500_list[i],start_date="2007-01-01",end_date="2017-12-31",authtoken='zpwaVHitms1rrbb-5sGs')
   except:
       error_list.append(sp500_list[i])

sp500_list=save_sp500_tickers()
#FB and NEM
data1=megadata['MSFT']['Close']
data2=megadata['NEM']['Close']
data3=megadata['GE']['Close']
data4=megadata['HOG']['Close']
data5=megadata['M']['Close']
data6=megadata['NVDA']['Close']
data7=megadata['FL']['Close']
data8=megadata['CTL']['Close']
data9=megadata['GOOGL']['Close']
data10=megadata['NRG']['Close']
data11=megadata['BIIB']['Close']
data12=megadata['AIZ']['Close']
data13=megadata['AZO']['Close']
data14=megadata['HST']['Close']
data15=megadata['EQIX']['Close']
data16=megadata['WMT']['Close']
data17=megadata['WYNN']['Close']
data18=megadata['EXPE']['Close']
data19=megadata['AMG']['Close']
data20=megadata['RCL']['Close']
# calculate miu
ininum=0
iniprice1=0
iniprice2=0
iniprice3=0
data12dic={}
data34dic={}
data56dic={}
data78dic={}
data910dic={}
data1112dic={}
data1314dic={}
data1516dic={}
data1718dic={}
data1920dic={}
datalist=[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19,data20]
diclist=[data12dic,data34dic,data56dic,data78dic,data910dic,data1112dic,data1314dic,data1516dic,data1718dic,data1920dic]
for k in range(int(len(datalist)/2)):
    for i in range(len(data1)):
        if i==len(data1)-1:
            iniprice1=iniprice1+datalist[2*k][i]
            iniprice2=iniprice2+datalist[2*k+1][i]
            ininum=ininum+1
            averageprice1=iniprice1/ininum
            averageprice2=iniprice2/ininum
            comblist=[averageprice1,averageprice2]
            keyname=str(data1.index[i])[:7]
            diclist[k][keyname]=comblist
            iniprice1=0
            iniprice2=0
            ininum=0
            break
        if str(data1.index[i])[5:7]==str(data1.index[i+1])[5:7]:
            ininum=ininum+1
            iniprice1=iniprice1+datalist[2*k][i]
            iniprice2=iniprice2+datalist[2*k+1][i]
        else:
            ininum=ininum+1
            iniprice1=iniprice1+datalist[2*k][i]
            iniprice2=iniprice2+datalist[2*k+1][i]
            averageprice1=iniprice1/ininum
            averageprice2=iniprice2/ininum
            comblist=[averageprice1,averageprice2]
            keyname=str(data1.index[i])[:7]
            diclist[k][keyname]=comblist
            ininum=0
            iniprice1=0
            iniprice2=0
factor1=[]
factor2=[]
factor3=[]
factor4=[]
factor5=[]
factor6=[]
returnof20=[]
#calculate factors
def factor_construction(r1,r2):
    return1=[]
    return2=[]
    timelist=list(data1.index)
    timelist1=timelist.copy()
    del(timelist[0])
    for i in range(len(data1)):
        if i !=len(data1)-1:
            a=r1[i+1]/r1[i]-1
            return1.append(a)
            b=r2[i+1]/r2[i]-1
            return2.append(b)
        else:
            break
    returnof20.append(return1)
    returnof20.append(return2)
    return1ser=pd.Series(return1,timelist)
    return2ser=pd.Series(return2,timelist)
#construct factors
    for i in range(10,len(return2ser)):
        templist1=[]
        templist2=[]
        for j in range(i-10,i):
            templist1.append(return1ser[j])
            templist2.append(return2ser[j])
        av1=np.average(templist1)
        av2=np.average(templist2)
        diff1=return1ser[i]-av1
        diff2=return2ser[i]-av2
        factor1.append(diff1**2)
        factor2.append(diff1**3)
        factor3.append(diff1**4)
        factor4.append(diff2**2)
        factor5.append(diff2**3)
        factor6.append(diff2**4)
for ss in range(int(len(datalist)/2)):
    factor_construction(datalist[ss*2],datalist[ss*2+1])
#calculate y
signal=[]
for k in range(len(diclist)):
    for i in range(10,len(returnof20[0])):
        origin_value=0.5*returnof20[2*k][i]+0.5*returnof20[2*k+1][i]
        new_value=0.6*returnof20[2*k][i]+0.4*returnof20[2*k+1][i]
        if new_value>origin_value:
            signal.append(1)
        else:
            signal.append(0)
array_x=preprocessing.scale(np.array([factor1,factor2,factor3,factor4,factor5,factor6]).transpose())   
array_y=np.array(signal)
#not shuffle
train_array_x=array_x[:20000]
train_array_y=array_y[:20000]
test_array_x=array_x[20000:]
test_array_y=array_y[20000:]
clf1=LogisticRegression()
clf2=SVC()
clf3=RandomForestClassifier()
clf4=GaussianNB()
clf5=tree.DecisionTreeClassifier()
clf6=KNeighborsClassifier()
clf1.fit(train_array_x,train_array_y)
clf2.fit(train_array_x,train_array_y)
clf3.fit(train_array_x,train_array_y)
clf4.fit(train_array_x,train_array_y)
clf5.fit(train_array_x,train_array_y)
clf6.fit(train_array_x,train_array_y)
y_predict1=clf1.predict(test_array_x)
y_predict2=clf2.predict(test_array_x)
y_predict3=clf3.predict(test_array_x)
y_predict4=clf4.predict(test_array_x)
y_predict5=clf5.predict(test_array_x)
y_predict6=clf6.predict(test_array_x)
score1=accuracy_score(test_array_y,y_predict1)
score2=accuracy_score(test_array_y,y_predict2)
score3=accuracy_score(test_array_y,y_predict3)
score4=accuracy_score(test_array_y,y_predict4)
score5=accuracy_score(test_array_y,y_predict5)
score6=accuracy_score(test_array_y,y_predict6)
#following part 1.keras(dl)2.performance
prediction_list=[y_predict1,y_predict2,y_predict3,y_predict4,y_predict5,y_predict6]
#select the last two pairs to test the effect
totalvalue1=[]
totalvalue2=[]
for i in range(len(prediction_list)):
    sonvalue1=[]
    sonvalue2=[]
    selected_signal1=prediction_list[i][len(prediction_list[i])-5514:len(prediction_list[i])-2757]
    selected_signal2=prediction_list[i][len(prediction_list[i])-2757:len(prediction_list[i])]
    weight11=0.5
    weight12=0.5
    inivalue1=returnof20[16][0]*weight11+returnof20[17][0]*(1-weight11)
    inivalue2=returnof20[18][0]*weight12+returnof20[19][0]*(1-weight12)
    sonvalue1.append(inivalue1)
    sonvalue2.append(inivalue2)
    for w in range(1,len(selected_signal1)):
        if selected_signal1[w]==1:
            weight11=weight11+0.05
        else:
            weight11=weight11-0.05
        if selected_signal2[w]==1:
            weight12=weight12+0.05
        else:
            weight12=weight12-0.05
        inivalue1=inivalue1+returnof20[16][w]*weight11+returnof20[17][w]*(1-weight11)
        inivalue2=inivalue2+returnof20[18][w]*weight12+returnof20[19][w]*(1-weight12)
        sonvalue1.append(inivalue1)
        sonvalue2.append(inivalue2)
    totalvalue1.append(sonvalue1)
    totalvalue2.append(sonvalue2)
benchmark1=[]
benchmark2=[]
for ww in range(len(returnof20[0][10:])):
    benchmark1.append(0.5*returnof20[16][10:][ww]+0.5*returnof20[17][10:][ww])
    benchmark2.append(0.5*returnof20[18][10:][ww]+0.5*returnof20[19][10:][ww])
"""
timelist=list(data1.index)
ref_timelist=[]
for i in range(11,len(timelist)):
    ref_timelist.append(str(timelist[i])[:9])
plt.plot(ref_timelist,benchmark1)
plt.plot(ref_timelist,totalvalue1[1])
"""
betalist=[]
for i in range(len(totalvalue1)):
    covariance=np.cov(totalvalue1[i],benchmark1)[0][1]
    variance=np.var(benchmark1)
    beta=covariance/variance
    betalist.append(beta)
for j in range(len(totalvalue2)):
    covariance=np.cov(totalvalue2[j],benchmark2)[0][1]
    variance=np.var(benchmark2)
    beta=covariance/variance
    betalist.append(beta)
alphalist=[]
for i in range(len(totalvalue1)):
    alphaelelist=[]
    for j in range(len(totalvalue1[i])):
        alphaelelist.append(totalvalue1[i][j]-betalist[i]*benchmark1[j])
    alphalist.append(alphaelelist)
for i in range(len(totalvalue2)):
    alphaelelist=[]
    for j in range(len(totalvalue2[i])):
        alphaelelist.append(totalvalue2[i][j]-betalist[i+6]*benchmark2[j])
    alphalist.append(alphaelelist)
    
    






















