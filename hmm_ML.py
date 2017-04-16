# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 18:05:39 2017

@author: czy
"""
import hmmlearn
from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
import os
import seaborn as sns
import time 

time1=time.time()
#initialize
sns.set_style('white')   #这个语句可以防止图标挡住原图
os.chdir(r"C:\Users\czy\Desktop\课程\topics")
data=pd.read_csv('indicators.csv')
data=data.set_index('date')
data_test=data[3000:]
data=data[:3000]
volume=data["volume"].values
close = data['close'].values
open_price=data['open'].values
high = data['high'].values
low = data['low'].values
swing=data['swing'].values[5:]
boll=data['boll'].values[5:]
cci=data['cci'].values[5:]
dma=data['dma'].values[5:]
kdj=data['kdj'].values[5:]
mtm=data['mtm'].values[5:]
macd=data['macd'].values[5:]
sar=data['sar'].values[5:]


logreturn = (np.log(np.array(close[1:]))-np.log(np.array(close[:-1])))[4:]
logreturn5 = np.log(np.array(close[5:]))-np.log(np.array(close[:-5]))
logvolume=(np.log(np.array(volume[1:]))-np.log(np.array(volume[:-1])))[4:]
diffreturn = (np.array(high)-np.array(low))/np.array(close)       
diffreturn=diffreturn[5:]

#hmm
X = np.column_stack([swing,logreturn5])
hmm = GaussianHMM(n_components = 18, covariance_type='diag',n_iter = 5000).fit(X)
latent_states_sequence = hmm.predict(X)

datelist = pd.to_datetime(data['close'].index[5:])
closeidx = close[5:]

#plot market state sequence
sns.set_style('white')
plt.figure(figsize = (13, 8))
for i in range(hmm.n_components):
    state = (latent_states_sequence == i)
    plt.plot(datelist[state],closeidx[state],'.',label = 'latent state %d'%i,lw = 1)
    plt.legend()
    plt.grid(1)

#identify latent state
data = pd.DataFrame({'datelist':datelist,'logreturn':logreturn,'state':latent_states_sequence}).set_index('datelist')

plt.figure(figsize=(13,8))
for i in range(hmm.n_components):
    state = (latent_states_sequence == i)
    idx = np.append(0,state[:-1])
    data['state %d_return'%i] = data.logreturn.multiply(idx,axis = 0) 
    plt.plot(np.exp(data['state %d_return' %i].cumsum()),label = 'latent_state %d'%i) 
    plt.legend()
    plt.grid(1)

#construct strategy
buy=0;
sell=0;
for i in range(hmm.n_components):
    temp=np.exp(data['state %d_return' %i].cumsum())
    if temp[-1]>1.2 :
        buy=buy+(latent_states_sequence == i)
    if temp[-1]<1 :
        sell=sell+(latent_states_sequence == i)
buy = np.append(0,buy[:-1])
sell = np.append(0,sell[:-1])
data['backtest_return'] = data.logreturn.multiply(buy,axis = 0) \
                             - data.logreturn.multiply(sell,axis = 0)
plt.figure(figsize = (13,8))
plt.plot_date(datelist,np.exp(data['backtest_return'].cumsum()),'-',label='backtest result')
plt.legend()
plt.grid(1)


#calculate retracement
backtest=np.exp(data['backtest_return'].cumsum())
retracement=0;
high=0;
for i in range(len(backtest)):
    if backtest[i]>high:
        high=backtest[i]
    temp=1-backtest[i]/high
    if temp>retracement:
        retracement=temp


#test hmm
volume=data_test["volume"].values
close = data_test['close'].values
open_price=data_test['open'].values
high = data_test['high'].values
low = data_test['low'].values
swing=data_test['swing'].values[5:]
boll=data_test['boll'].values[5:]
cci=data_test['cci'].values[5:]
dma=data_test['dma'].values[5:]
kdj=data_test['kdj'].values[5:]
mtm=data_test['mtm'].values[5:]
macd=data_test['macd'].values[5:]
sar=data_test['sar'].values[5:]

logreturn = (np.log(np.array(close[1:]))-np.log(np.array(close[:-1])))[4:]
logreturn5 = np.log(np.array(close[5:]))-np.log(np.array(close[:-5]))
diffreturn = (np.array(high)-np.array(low))/np.array(close)       
diffreturn=diffreturn[5:]



X = np.column_stack([swing,logreturn5])
latent_states_sequence = hmm.predict(X)

datelist = pd.to_datetime(data_test['close'].index[5:])
closeidx = close[5:]

sns.set_style('white')
plt.figure(figsize = (13, 8))
for i in range(hmm.n_components):
    state = (latent_states_sequence == i)
    plt.plot(datelist[state],closeidx[state],'.',label = 'latent state %d'%i,lw = 1)
    plt.legend()
    plt.grid(1)

data_test = pd.DataFrame({'datelist':datelist,'logreturn':logreturn,'state':latent_states_sequence}).set_index('datelist')

plt.figure(figsize=(13,8))
for i in range(hmm.n_components):
    state = (latent_states_sequence == i)
    idx = np.append(0,state[:-1])
    data_test['state %d_return'%i] = data_test.logreturn.multiply(idx,axis = 0) 
    plt.plot(np.exp(data_test['state %d_return' %i].cumsum()),label = 'latent_state %d'%i) 
    plt.legend()
    plt.grid(1)
    
buy=0
sell=0
for i in range(hmm.n_components):
    temp=np.exp(data_test['state %d_return' %i].cumsum())
    if temp[-1]>1.14 :
        buy=buy+(latent_states_sequence == i)
    if temp[-1]<0.99:
        sell=sell+(latent_states_sequence == i)
buy = np.append(0,buy[:-1])
sell = np.append(0,sell[:-1])
data_test['backtest_return'] = data_test.logreturn.multiply(buy,axis = 0) \
                             - data_test.logreturn.multiply(sell,axis = 0)
plt.figure(figsize = (13,8))
plt.plot_date(datelist,np.exp(data_test['backtest_return'].cumsum()),'-',label='backtest result')
plt.legend()
plt.grid(1)    

#retracement
backtest=np.exp(data_test['backtest_return'].cumsum())
retracement=0
high=0
for i in range(len(backtest)):
    if backtest[i]>high:
        high=backtest[i]
    temp=1-backtest[i]/high
    if temp>retracement:
        retracement=temp    
print(retracement)

#return to retracement
print(backtest[-1])
print(backtest[-1]/retracement)

#success rate
predict_index=buy-sell
success=0
for i in range(len(predict_index)):
    if (logreturn[i]*predict_index[i]>=0):
        success=success+1;
success_rate=success/len(predict_index)
print(success_rate)

time2=time.time()
print(time2-time1)

#three signal graph
sns.set_style('white')
plt.figure(figsize = (13, 8))
for i in range(-1,2):
    state = (predict_index == i)
    if i==1:
        plt.plot(datelist[state],closeidx[state],'.',label = 'up',lw = 1)
    elif i==0:
        plt.plot(datelist[state],closeidx[state],'.',label = 'shock',lw = 1)
    else:
        plt.plot(datelist[state],closeidx[state],'.',label = 'down',lw = 1)
    plt.legend()
    plt.grid(1)

                

