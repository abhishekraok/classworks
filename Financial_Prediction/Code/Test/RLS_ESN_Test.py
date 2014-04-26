# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 14:38:43 2014

@author: Abhishek
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
if not sys.path.count(".."): sys.path.append("..")
import ESNFile as es

X = np.loadtxt('../data/MackeyGlass_t17.txt')[:800]
ytest = X[1:]
X = X[:-1]
mod1 = es.ESN(resSize=4,)
mod1,yp = mod1.adaptfitpredict(X,ytest)
aWout = mod1.Wout
print (yp-ytest).var()/ytest.var()
plt.figure()
plt.plot(yp,'o-',label='Adapt Predicted')
plt.plot(ytest,'d--',label='Test target')

#yp = mod1.fit(X,ytest).predict(X)
#plt.plot(yp,'o-',label='fit Predicted')
#fitWout = mod1.Wout
#print (yp-ytest).var()/ytest.var()
#print aWout,fitWout
plt.legend()