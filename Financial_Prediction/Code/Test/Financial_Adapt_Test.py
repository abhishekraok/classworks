# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 23:10:40 2014

@author: Abhishek

Financial prediction using adaptive ESN
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
if not sys.path.count(".."): sys.path.append("..")
import ESNFile as es

# Data handling
initLen = 100
modelsets = [es.ESN(resSize=99, initLen=initLen)]
print 'Initializing done'
modelnames = ['ESN']

datasets = [pd.read_csv('../data/pdeqretsnonan.csv',
                 index_col=0, parse_dates=True).ix[:,:5].resample('W',
                 how='mean').values]
datasetnames = ['Financial']

for X,dname in zip(*[datasets,datasetnames]):
    y = X[1:]
    X = X[:-1]
    for ithmodel, mname in zip(*[modelsets,modelnames]):  
        model = ithmodel
        yp = model.adaptfitpredict(X,y)
        y = y[initLen:]
        NMSE = (yp-y).var() / y.var()
        print 'The NMSE for model {0} using data {1} is {2}'.format(mname,dname,NMSE)
#plt.figure()
#plot(X,)
#plt.title('The Entire Time Series')
plt.figure()
plot(y[:,0],label='Test Target')
plot(yp[:,0],label='Predicted')
plt.title('Predicted vs actual')
plt.legend()
