# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:59:15 2014

@author: Colleen
"""

import numpy as np
import matplotlib.pyplot as plt
import ESNFile as es

X = np.loadtxt('data/EEG_EYE.txt', delimiter = ',')
X = X[:,1].flatten()
X = X - X.mean()
Xtrain,Xtest,ytrain,ytest = es.ts_train_test_split(X)
mod1 = es.ESN()
mod1 = mod1.fit(Xtrain,ytrain)
yp = mod1.predict(Xtest)
NMSE = (yp - ytest).var() / ytest.var()
print NMSE
plt.figure()
plot(yp,label='Predicted values')
plot(ytest,label='Actual Values')
plt.legend()