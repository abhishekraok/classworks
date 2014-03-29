# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 17:01:22 2014

@author: Abhishek

Testing sunspots with SVR model
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from sklearn.svm import SVR

def getdata():
    """ get the sunpsots data"""
    dta = sm.datasets.sunspots.load_pandas().data
    dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
    del dta["YEAR"]
    return dta

s = getdata()
ORDER = 3 # Order of prediction
X = pd.concat([s.shift(i) for i in range(1,ORDER+1)], axis=1).dropna()
X.columns = [s.columns[0]+ '_lag_'+ str(i+1) for i in range(X.shape[1])]
y = s.loc[X.head(1).index[0]:X.tail(1).index[0]]
Xtrain = X[:'1920']
ytrain = y[:'1920']
Xtest =  X['1920':'1988']
ytest = y['1920':'1988']      
gamma = 1/ (2*(290.19)**2)
clf = SVR(C=43.6432,kernel='rbf',epsilon=0.5,gamma=gamma)
model = clf.fit(Xtrain.values,ytrain.values.flatten())
yp = pd.DataFrame( data=model.predict(Xtest), 
                  index=ytest.index, columns=ytest.columns)
NMSE = (yp - ytest).var() / ytest.var()
print 'The NMSE is '
print NMSE

ax = ytest.plot()
ax = yp.plot(ax=ax,style = 'd-', label='Predicted')
ax.legend()

