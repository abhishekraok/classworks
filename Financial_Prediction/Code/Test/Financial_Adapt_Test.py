# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 23:10:40 2014

@author: Abhishek

Financial prediction using adaptive ESN
"""
import pandas as pd
import matplotlib.pyplot as plt
import sys
if not sys.path.count(".."): sys.path.append("..")
import ESNFile as es
from sklearn.preprocessing import normalize

# Data handling
initLen = 100
modelsets = [es.ESNAdapt(resSize=199, initLen=initLen)]
print 'Initializing done'
modelnames = ['ESN']

datasets = [pd.read_csv('../data/pdeqretsnonan.csv',
                 index_col=0, parse_dates=True).ix[:,:5].resample('M',
                 how='mean').values]
datasetnames = ['Financial']

for X,dname in zip(*[datasets,datasetnames]):
    X = normalize(X)
    y = X[1:]
    X = X[:-1]
    for ithmodel, mname in zip(*[modelsets,modelnames]):  
        model = ithmodel
        yp = model.predict(X,y)
        y = y[initLen:]
        NMSE = (yp-y).var() / y.var()
        print 'The NMSE for model {0} using data {1} is {2}'.format(mname,dname,NMSE)
#plt.figure()
#plot(X,)
#plt.title('The Entire Time Series')
plt.figure()
plt.plot(y[:200,0],label='Test Target')
plt.plot(yp[:200,0],'o--',label='Predicted')
plt.title('Predicted vs actual')
plt.legend()
