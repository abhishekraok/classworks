# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 16:48:37 2014

@author: Colleen
"""
import sys
if not sys.path.count(".."): sys.path.append("..")
import ESNFile as es
import ESNModified as esm
import numpy as np
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
#from Spline import readGaitData

modelsets = [es.ESNC(initLen = 0),SVC(),esm.ESN(initLen = 0)]
modelnames = ['ESNC','SVC','ESN hard limiter']

XActual = np.loadtxt('../data/Gait1.txt') #**
y = XActual[::10,-1]           #y is the last column of data set
X = XActual[::10,1:-1]          #X is all of the data except last column
X = preprocessing.normalize(X)
Xtrain, Xtest, ytrain, ytest = X[::2], X[1::2], y[::2], y[1::2]

ResultTable = pd.DataFrame(columns=['Model','Data','Accuracy','Precision', 'Recall'])
print 'Initialization done'

for ithmodel, mname in zip(*[modelsets,modelnames]):  
    model = ithmodel
    yp = model.fit(Xtrain,ytrain).predict(Xtest)
    ytestFreeze = np.zeros(ytest[ytest > 0].shape[0])
    ypFreeze = np.zeros(ytest[ytest > 0].shape[0])
    for i in range(yp.shape[0]):
        count = 0
        if yp[i] == 2:
            ypFreeze[count] = 1
            if ytest[i] > 0:
                ytestFreeze[count] = ytest[i] - 1
            count = count + 1
        elif yp[i] == 1:
            ypFreeze[count] = 0
            if ytest[i] > 0:
                ytestFreeze[count] = ytest[i] - 1
            count = count + 1
    #ypfreeze = [1 if i==2 else 0 for i in yp[yp>0]]
    #ytestfreeze = [1 if i==2 else 0 for i in ytest[ytest>0]]
    Accuracy = np.average([i==j for i,j in zip(yp,ytest)])
    Precision = precision_score(ytestFreeze, ypFreeze)
    Recall = recall_score(ytestFreeze, ypFreeze)
    res = {'Model':mname,'Data':'Gait','Accuracy':Accuracy, 'Precision':Precision, 'Recall': Recall}
    ResultTable = ResultTable.append(res,ignore_index=True)   
    print res 
    plt.figure()
    plt.title(mname+' Gait')
    plot(ytest,label='Actual')
    plot(yp,label='Predicted')
    plt.legend()
    plt.axis([0,ytest.shape[0],-.3,2.3])
        
print ResultTable
#ResultTable.to_csv('ComparisonRes.csv')