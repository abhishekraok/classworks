# -*- coding: utf-8 -*-
"""
Created on Tue Apr 08 19:38:20 2014

@author: Abhishek Rao

Compare Echo State Network (ESN) time series prediction with LPC and SVR.
Data set used is sunspots and Financial data.
Metric used is Normalized Mean Square Error (NMSE).
"""

import ESNFile as es
import numpy as np
import statsmodels.api as sm
from sklearn.svm import SVR
import pandas as pd
from sklearn import linear_model

svmd = SVR(C=43.6432,kernel='rbf',epsilon=0.5,gamma=1/ (2*(290.19)**2))  
modelsets = [es.ESN(),svmd, linear_model.Ridge ()]
modelnames = ['SVR','ESN','Linear Ridge Regression']

datasets = [sm.datasets.sunspots.load_pandas().data.ix[:,1].values,
            np.loadtxt('data/MackeyGlass_t17.txt'),
            pd.read_csv('data/pdeqretsnonan.csv',
                 index_col=0, parse_dates=True).ix[:,5].resample('M',
                 how='mean').values]
datasetnames = ['Sunspots','MackayGlass','Financial']
outputstring = ''

for X,dname in zip(*[datasets,datasetnames]):
    Xtrain,Xtest,ytrain,ytest = es.ts_train_test_split(X, lags=3)
    for ithmodel, mname in zip(*[modelsets,modelnames]):  
        model = ithmodel
        model = ithmodel.fit(Xtrain,ytrain)
        yp = model.predict(Xtest)
        NMSE = (yp-ytest).var() / ytest.var()
        outputstring += 'The NMSE for model {0} using data {1} is {2}\n'.format(mname,dname,NMSE)
print outputstring
open('Model_Compare.txt','w').write(outputstring)