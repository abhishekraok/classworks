# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:59:15 2014

@author: Colleen
"""

import numpy as np
import matplotlib.pyplot as plt
import ESNModified as es
from sklearn import preprocessing

X = np.loadtxt('data/EEG_EYE.txt', delimiter = ',')
y = X[:,-1]           #y is the last column of data set
X = X[:,:-1]          #X is all of the data except last column
X = preprocessing.normalize(X)


Xtrain, Xtest, a, b = es.ts_train_test_split(X)     #split X into Xtrain and Xtest
ytrain, ytest, c, d = es.ts_train_test_split(y)     #split Y into ytrain and ytest

mod1 = es.ESN()                                     #mod1 is an ESN
mod1 = mod1.fit(Xtrain,ytrain)                      #fit model to training set
yp = mod1.predict(Xtest)                            #yp is predicted y value of Xtest data

NMSE = (yp - ytest).var() / ytest.var()
print NMSE

plt.figure()
plt.plot(yp[:999],'d-', label='Predicted values')
plt.plot(ytest[:999],'d-', label='Actual Values')
plt.legend()

print np.average(ytest != yp)

plt.figure()
plt.plot(yp, label='Predicted values')
plt.plot(ytest, label='Actual Values')
plt.legend()