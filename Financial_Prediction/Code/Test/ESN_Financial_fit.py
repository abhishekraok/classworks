# -*- coding: utf-8 -*-
"""
Created on Sat Apr 05 22:40:06 2014

@author: Abhishek

Financial prediction using ESN
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ESNFile as es

# Data handling
#df = pd.read_csv('../data/pdeqretsnonan.csv',index_col=0, parse_dates=True)
#X = df.ix[:,:5].resample('W', how='mean')

modelsets = [es.ESN(resSize=999)]
modelnames = ['ESN']

datasets = [pd.read_csv('../data/pdeqretsnonan.csv',
                 index_col=0, parse_dates=True).ix[:,:5].resample('M',
                 how='mean').values]
datasetnames = ['Financial']

for X,dname in zip(*[datasets,datasetnames]):
    Xtrain,Xtest,ytrain,ytest = es.ts_train_test_split(X, lags=3)
    for ithmodel, mname in zip(*[modelsets,modelnames]):  
        model = ithmodel
        model = ithmodel.fit(Xtrain,ytrain)
        yp = model.predict(Xtest)
        NMSE = (yp-ytest).var() / ytest.var()
        print 'The NMSE for model {0} using data {1} is {2}'.format(mname,dname,NMSE)
#plt.figure()
#plot(X,)
#plt.title('The Entire Time Series')
plt.figure()
plot(ytest,label='Test Target')
plot(yp,'o--',label='Predicted')
plt.title('Predicted vs actual')
plt.legend()

#y = df['Company_21'].resample('M', how='mean')
#y=y[1:]
#X=X[:-1]
#
#rtt = 0.8 # Ratio of train to test length
#trainLen = int(np.round(X.shape[0]*rtt))
#Xtrain = X[:trainLen]
#ytrain = y[:trainLen]
#Xtest =  X[trainLen:]
#ytest = y[trainLen:]      
#
##Machine learning part begins 
#clf = es.ESN(initLen=20, resSize = 999)
#model = clf.fit(Xtrain.values[:,None],ytrain.values[:,None])
#yp = pd.DataFrame( data=model.predict(Xtest.values), 
#                  index=ytest.index, ) #columns=ytest.columns
#NMSE = (yp - ytest).var() / ytest.var()
#print 'The NMSE is '
#print NMSE
#
#ax = ytest.plot(style='o--')
#ax = yp.plot(ax=ax,style = 'd-', label='Predicted')
#ax.legend()
#mod1 = es.ESN(resSize=999,initLen=100)
#NMSE,yp,ytest = mod1.checkts(X.values)
#plt.figure()
#plt.plot(yp[:999],'o-',label='Predicted')
#plt.plot(ytest[:999],'^-.',label='Test target')
#plt.legend()
#plt.title('Prediction for reservoir size {0}'.format(i))
#print 'For reservoir size {0} the NMSE is {1}'.format(i,NMSE) 