# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 17:01:22 2014

@author: Abhishek

Testing sunspots with ARMA model
"""
import numpy as np
from scipy import stats
import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

def arfx(dta):
    # Fit AR9, ARMA2, ARMA3 models and check
    ar_mod = sm.tsa.AR(dta['1700':'1920'])
    ar_results= ar_mod.fit(order=9)
    mod9_predict = ar_results.predict('1920','1988',dynamic=True)
    arma_mod20 = sm.tsa.ARMA(dta, (2,0)).fit()
    mod20_predict = arma_mod20.predict('1920','1988',dynamic=True)
    arma_mod30 = sm.tsa.ARMA(dta, (3,0)).fit()
    mod30_predict = arma_mod30.predict('1920','1988',dynamic=True)
    
    ax = dta.ix['1920':'1988'].plot()
    ax = mod9_predict.plot(ax=ax, style='r--', label='AR 9 model');
    ax = mod20_predict.plot(ax=ax, style='g--', label='ARMA 2');
    ax = mod30_predict.plot(ax=ax, style='o', label='ARMA 3');

def simplepred(dta):
    """ Simple predictor where yp = x[n-1]"""
    return dta.shift(-1)

def measMSE(predicted, target):
    """ calculates the MSE between predicted and target"""
    dif = target - predicted
    return (dif**2).sum()/dif.shape[0]

dta = sm.datasets.sunspots.load_pandas().data
dta.index = pandas.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]

Xtrain = dta['1700':'1920']
Xtest =  dta['1920':'1988']
     
yp = simplepred(Xtest) 
MSE = measMSE(yp,Xtest)
print 'The MSE is '
print MSE

ax = yp.plot(style = 'd')
ax = Xtest.plot(ax=ax)

