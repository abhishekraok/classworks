# -*- coding: utf-8 -*-
"""
Created on Tue Apr 08 19:38:20 2014

@author: Abhishek Rao

Compare the performance of ESNC, ESN with hard limiter and SVC for 
EEG Eye data
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


modelsets = [es.ESNC(initLen = 0), SVC(), esm.ESN(initLen = 0)]
modelnames = ['ESNC','SVC','ESN hard limiter']

datasets = [np.loadtxt('../data/EEG_EYE.txt', delimiter = ',')]
datasetnames = ['EEG_Eye']
ResultTable = pd.DataFrame(columns=['Model','Data','Accuracy', 'Precision', 'Recall'])
print 'Initialization done'

for X,dname in zip(*[datasets,datasetnames]):
    y = X[:,-1]           #y is the last column of data set
    X = X[:,:-1]          #X is all of the data except last column
    X = preprocessing.normalize(X)
#    train_ratio = 0.7     #Ratio of training set / full set
#    Ntrain = int(X.shape[0]*train_ratio)   #Number of data points
    #Xtrain,Xtest,ytrain,ytest =train_test_split(X,y) #X[:Ntrain],X[Ntrain:], y[:Ntrain],y[Ntrain:]
    Xtrain,Xtest,ytrain,ytest =X[::2],X[1::2], y[::2],y[1::2]
    for ithmodel, mname in zip(*[modelsets,modelnames]):  
        model = ithmodel
        yp = model.fit(Xtrain,ytrain).predict(Xtest)
        Accuracy = np.average([i==j for i,j in zip(yp,ytest)])
        Precision = precision_score(ytest, yp)
        Recall = recall_score(ytest, yp)
        res = {'Model':mname,'Data':dname,'Accuracy':Accuracy, 'Precision':Precision, 'Recall': Recall}
        ResultTable = ResultTable.append(res,ignore_index=True)   
        print res 
        plt.figure()
        plt.title(mname+' '+dname)
        plot(ytest,label='Actual')
        plot(yp,label='Predicted')
        plt.legend()
        plt.axis([0,ytest.shape[0],-.3,1.3])
        
print ResultTable
#ResultTable.to_csv('ComparisonRes.csv')