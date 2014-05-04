# -*- coding: utf-8 -*-
"""
Created on Tue Apr 08 19:38:20 2014

@author: Abhishek Rao

Compare the performance of ESNC, ESN with hard limiter and SVC for 
EEG Eye data

Data from: http://archive.ics.uci.edu/ml/datasets/EEG+Eye+State#
"""
import sys
if not sys.path.count(".."): sys.path.append("..")
import ESNFile as es
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt


modelset = [es.ESNC(initLen=0), SVC(), es.ESNHardLimiter(initLen=0)]
modelnames = ['ESNC', 'SVC', 'ESN hard limiter']

#preserved structure to use multiple data sets in future
datasets = [np.loadtxt('../data/EEG_EYE.txt', delimiter = ',')]
datasetnames = ['EEG_Eye']
# ResultTable stores values of all comparison measures
ResultTable = pd.DataFrame(columns=['Model', 'Data', 'Accuracy', 
                                    'Precision', 'Recall'])
print 'Initialization done'

for X, dname in zip(*[datasets, datasetnames]):
    Y = X[:, -1]           #Y is the last column of data set
    X = X[:, :-1]          #X is all of the data except last column
    X = preprocessing.normalize(X)
    #Split the training and test set by using every other data point (50:50)
    Xtrain, Xtest, Ytrain, Ytest = X[::2], X[1::2], Y[::2], Y[1::2]
    
    for ithmodel, mname in zip(*[modelset, modelnames]):  
        model = ithmodel
        Ypredict = model.fit(Xtrain, Ytrain).predict(Xtest)  # fit then predict
        
        #calculate comparison measures
        accuracy = np.average([i == j for i, j in zip(Ypredict, Ytest)])
        precision = precision_score(Ytest, Ypredict)
        recall = recall_score(Ytest, Ypredict)
        results = {'Model': mname, 'Data': dname, 'Accuracy': accuracy, 
               'Precision': precision, 'Recall': recall}
        ResultTable = ResultTable.append(results, ignore_index=True)   
        print results
        
        # plot
        plt.figure()
        plt.title(mname+ ' ' + dname)
        plt.plot(Ytest, label='Actual')
        plt.plot(Ypredict, label='Predicted')
        plt.legend()
        plt.axis([0, Ytest.shape[0], -.3, 1.3])
        
print ResultTable