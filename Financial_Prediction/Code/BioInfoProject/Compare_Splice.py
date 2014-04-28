# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:57:19 2014

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
from Spline import readSpliceData

modelsets = [es.ESNC(initLen = 0),SVC(),esm.ESN(initLen = 0)]
modelnames = ['ESNC','SVC','ESN hard limiter']

X,y = readSpliceData('../data/splice.txt')
Xtrain, Xtest, ytrain, ytest = X[::2], X[1::2], y[::2], y[1::2]

ResultTable = pd.DataFrame(columns=['Model','Data','Accuracy','Precision', 'Recall'])
print 'Initialization done'


for ithmodel, mname in zip(*[modelsets,modelnames]):  
    model = ithmodel
    yp = model.fit(Xtrain,ytrain).predict(Xtest)
    Accuracy = np.average([i==j for i,j in zip(yp,ytest)])
    Precision = precision_score(ytest, yp, average = none)
    Recall = 
    res = {'Model':mname,'Data':'Splice','Accuracy':Accuracy,'Precision of ':Precision, 'Recall':Recall}
    ResultTable = ResultTable.append(res,ignore_index=True)   
    print res 
    plt.figure()
    plt.title(mname+' Splice')
    plot(ytest,label='Actual')
    plot(yp,label='Predicted')
    plt.legend()
    plt.axis([0,ytest.shape[0],-.3,2.3])
        
print ResultTable
#ResultTable.to_csv('ComparisonRes.csv')