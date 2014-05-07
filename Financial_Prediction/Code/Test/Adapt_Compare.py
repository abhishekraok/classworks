# -*- coding: utf-8 -*-
"""
Created on Tue May 06 21:16:12 2014

@author: Abhishek

Data prediction using adaptive ESN. Parameter optimization.
"""
import pandas as pd
import sys
if '..' not in sys.path: sys.path.append('..')
import ESNFile as es
from sklearn.preprocessing import normalize

initLen = 20
leak_rate_set = [0.01,0.1,0.3]
res_size_set = [20,100,200]
Fin_data = pd.read_csv('../data/pdeqretsnonan.csv',
                 index_col=0, parse_dates=True)
datasets = [Fin_data.ix[:,:50].resample('M',how='mean').values,
            Fin_data.ix[:,:5].resample('W',how='mean').values,
            Fin_data.ix[:,:2].values]
ResultTable = pd.DataFrame(columns=['leak_rate','resSize','Data','NMSE'])

datasetnames = ['Monthly','Weekly','Daily']
for alpha in leak_rate_set:
    for resSize in res_size_set:    
        for X,dname in zip(*[datasets,datasetnames]):
            y = X[1:]
            X = X[:-1]
            model = es.ESNAdapt(leakRate=alpha, resSize=resSize,
                                initLen=initLen)
            yp = model.predict(X,y)
            y = y[initLen:]
            NMSE = (yp-y).var() / y.var()
            res = {'leak_rate':alpha,'resSize':resSize,'Data':dname,'NMSE':("%.4f" %NMSE)}
            ResultTable = ResultTable.append(res,ignore_index=True)
            print res
print 'The final result is ',ResultTable