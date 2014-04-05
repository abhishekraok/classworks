# -*- coding: utf-8 -*-
"""
Created on Sat Apr 05 14:21:52 2014

@author: Abhishek

Testing sunspots with ESN model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import ESNFile as es

def getdata():
    """ get the sunpsots data"""
    dta = sm.datasets.sunspots.load_pandas().data
    dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
    del dta["YEAR"]
    return dta
    
# Data handling
X = getdata()
y = X.loc[X.head(1).index[0]:X.tail(1).index[0]]
Xtrain = X[:'1920']
ytrain = y[:'1920']
Xtest =  X['1920':'1988']
ytest = y['1920':'1988']      

#Machine learning part begins 
clf = es.ESN(initLen=20, resSize = 20)
model = clf.fit(Xtrain.values,ytrain.values)
yp = pd.DataFrame( data=model.predict(Xtest.values), 
                  index=ytest.index, columns=ytest.columns)
NMSE = (yp - ytest).var() / ytest.var()
print 'The NMSE is '
print NMSE

ax = ytest.plot()
ax = yp.plot(ax=ax,style = 'd-', label='Predicted')
ax.legend()
