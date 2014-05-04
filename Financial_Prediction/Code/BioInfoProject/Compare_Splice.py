# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:57:19 2014

@author: Colleen O'Rourke

Compare the performance of ESNC, ESN with hard limiter and SVC for 
Splice-junction Gene Sequence data

Data from: http://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Splice-
            junction+Gene+Sequences%29
"""

import sys
if not sys.path.count(".."): sys.path.append("..")
import ESNFile as es
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
# from sklearn.cross_validation import train_test_split  # to shuffle data
import matplotlib.pyplot as plt
from Spline import readSpliceData

modelset = [es.ESNC(initLen=0), SVC(), es.ESNHardLimiter(initLen=0)]
modelnames = ['ESNC', 'SVC', 'ESN hard limiter']

# converts labels into 0, 1, or 2 and sequences into array of integers
X, Y = readSpliceData('../data/splice.txt')
# Split the training and test set by using every other data point (50:50)
Xtrain, Xtest, Ytrain, Ytest = X[::2], X[1::2], Y[::2], Y[1::2]  
# (Alternatively, use this to shuffle the data:)
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.5,
#                                                           random_state = 42) 
# ResultTables stores all values of comparison measures
ResultTable = pd.DataFrame(columns=['Model', 'Data', 'Accuracy', 'ErrorEI', 
                                    'ErrorIE', 'ErrorN', 'Precision EI', 
                                    'Precision IE', 'Precision N', 'Recall EI',
                                    'Recall IE', 'Recall N'])
print 'Initialization done'

for ithmodel, mname in zip(*[modelset, modelnames]):  
    model = ithmodel
    Ypredict = model.fit(Xtrain,Ytrain).predict(Xtest)     # fit then predict
    
    # calculate comparison measures
    accuracy = np.average([i == j for i, j in zip(Ypredict, Ytest)])
    ErrorEI = np.average([i != j for i, j in zip(Ypredict[Ytest == 0], 
                                                 Ytest[Ytest == 0])])
    ErrorIE = np.average([i != j for i, j in zip(Ypredict[Ytest == 1], 
                                                 Ytest[Ytest == 1])])
    ErrorN  = np.average([i != j for i, j in zip(Ypredict[Ytest == 2], 
                                                 Ytest[Ytest == 2])])
    precision = precision_score(Ytest, Ypredict, average = None)
    recall = recall_score(Ytest, Ypredict, average = None)
    results = {'Model': mname, 'Data': 'Splice', 'Accuracy': accuracy, 
               'ErrorEI': ErrorEI, 'ErrorIE': ErrorIE, 'ErrorN': ErrorN, 
               'Precision EI': precision[0], 'Precision IE': precision[1], 
               'Precision N': precision[2],'Recall EI': recall[0], 
               'Recall IE': recall[1], 'Recall N': recall[2]}
    ResultTable = ResultTable.append(results, ignore_index=True)   
    print results
    
    # plot
    plt.figure()
    plt.title(mname + ' Splice')
    plt.plot(Ytest, label='Actual')
    plt.plot(Ypredict, label='Predicted')
    plt.legend()
    plt.axis([0, Ytest.shape[0], -.3, 2.3])
        
print ResultTable