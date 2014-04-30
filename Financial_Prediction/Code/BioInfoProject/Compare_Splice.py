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
Xtrain, Xtest, ytrain, ytest = X[::2], X[1::2], y[::2], y[1::2]     #use every other data point

ResultTable = pd.DataFrame(columns=['Model','Data','Accuracy','ErrorEI','ErrorIE','ErrorN','Precision EI', 'Precision IE', 'Precision N','Recall EI', 'Recall IE', 'Recall N'])
print 'Initialization done'


for ithmodel, mname in zip(*[modelsets,modelnames]):  
    model = ithmodel
    yp = model.fit(Xtrain,ytrain).predict(Xtest)
    #Calculate effectiveness
    Accuracy = np.average([i==j for i,j in zip(yp,ytest)])
    ErrorEI = np.average([i!=j for i,j in zip(yp[ytest == 0],ytest[ytest == 0])])
    ErrorIE = np.average([i!=j for i,j in zip(yp[ytest == 1],ytest[ytest == 1])])
    ErrorN = np.average([i!=j for i,j in zip(yp[ytest == 2],ytest[ytest == 2])])
    precision = precision_score(ytest, yp, average = None)
    PrecEI = precision[0]
    PrecIE = precision[1]
    PrecN = precision[2]
    recall = recall_score(ytest, yp, average = None)
    RecEI = recall[0]
    RecIE = recall[1]
    RecN = recall[2]
    res = {'Model':mname,'Data':'Splice','Accuracy':Accuracy,'ErrorEI':ErrorEI,'ErrorIE':ErrorIE,'ErrorN':ErrorN,'Precision EI':PrecEI, 'Precision IE':PrecIE, 'Precision N':PrecN,'Recall EI':RecEI, 'Recall IE':RecIE, 'Recall N':RecN}
    ResultTable = ResultTable.append(res,ignore_index=True)   
    print res 
    #Plot
    plt.figure()
    plt.title(mname+' Splice')
    plot(ytest,label='Actual')
    plot(yp,label='Predicted')
    plt.legend()
    plt.axis([0,ytest.shape[0],-.3,2.3])
        
print ResultTable
#ResultTable.to_csv('ComparisonRes.csv')