# -*- coding: utf-8 -*-
"""
Created on Tue Apr 08 19:38:20 2014

@author: Abhishek Rao

Compare Echo State Network (ESN) time series prediction with LPC
Linear Ridge Regression (LRR) and Support Vector Regression (SVR).
Data set used is sunspots and Financial data.
Metric used is Normalized Mean Square Error (NMSE).
"""

import ESNFile as es
import numpy as np
import statsmodels.api as sm
from sklearn.svm import SVR
import pandas as pd
from sklearn import linear_model

modelsets = [es.ESN(),SVR(), linear_model.Ridge ()]
modelnames = ['ESN','SVR','LRR']

datasets = [sm.datasets.sunspots.load_pandas().data.ix[:,1].values,
            np.loadtxt('data/MackeyGlass_t17.txt'),
            pd.read_csv('data/pdeqretsnonan.csv',
                 index_col=0, parse_dates=True).ix[:,5].resample('M',
                 how='mean').values]
datasetnames = ['Sunspots','MackayGlass','Financial']
ResultTable = pd.DataFrame(columns=['Model','Data','NMSE'])

for X,dname in zip(*[datasets,datasetnames]):
    Xtrain,Xtest,ytrain,ytest = es.ts_train_test_split(X, lags=3)
    for ithmodel, mname in zip(*[modelsets,modelnames]):  
        model = ithmodel
        yp = model.fit(Xtrain,ytrain).predict(Xtest)
        NMSE = (yp-ytest).var() / ytest.var()
        res = {'Model':mname,'Data':dname,'NMSE':NMSE}
        ResultTable = ResultTable.append(res,ignore_index=True)               

print ResultTable
ResultTable.to_csv('ComparisonRes.csv')