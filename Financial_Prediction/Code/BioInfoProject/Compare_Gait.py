# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 16:48:37 2014

@author: Colleen O'Rourke

Compare the performance of ESNC, ESN with hard limiter and SVC for 
Freezing of Gait data

Data from: http://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait
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

x_input = np.loadtxt('../data/Gait1.txt')
Y = x_input[::10, -1]       # Y is the last column; every 10th point
X = x_input[::10, 1:-1]     # X is every 10th data point except last column
X = preprocessing.normalize(X)
#Split the training and test set by using every other data point (50:50)
Xtrain, Xtest, Ytrain, Ytest = X[::2], X[1::2], Y[::2], Y[1::2]                 
# ResultTable stores values of all comparison measures
ResultTable = pd.DataFrame(columns=['Model', 'Data', 'Accuracy', 
                                    'Precision', 'Recall'])
print 'Initialization done'

for ithmodel, mname in zip(*[modelset, modelnames]):  
    model = ithmodel
    Ypredict = model.fit(Xtrain,Ytrain).predict(Xtest)     # fit then predict
    
    # delete all 0's from Ytest and Ypredict; 
    # convert 1's and 2's into binary (0 and 1, respectively) 
    Ytest_binary = []
    Ypredict_binary = []
    for i in range(Ypredict.shape[0]):
        if Ypredict[i] == 2:
            Ypredict_binary.append(1)
            if Ytest[i] > 0:
                Ytest_binary.append(Ytest[i] - 1)
            else:
                Ytest_binary.append(0)
        elif Ypredict[i] == 1:
            Ypredict_binary.append(0)
            if Ytest[i] > 0:
                Ytest_binary.append(Ytest[i] - 1)
            else:
                Ytest_binary.append(0)
                
    #calculate comparison measures
    accuracy = np.average([i == j for i, j in zip(Ypredict, Ytest)])
    precision = precision_score(Ytest_binary, Ypredict_binary)
    recall = recall_score(Ytest_binary, Ypredict_binary)
    results = {'Model': mname, 'Data': 'Gait', 'Accuracy': accuracy, 
           'Precision': precision, 'Recall': recall}
    ResultTable = ResultTable.append(results, ignore_index=True)   
    print results
    
    # plot
    plt.figure()
    plt.title(mname + ' Gait')
    plt.plot(Ytest, label='Actual')
    plt.plot(Ypredict, label='Predicted')
    plt.legend()
    plt.axis([0, Ytest.shape[0], -.3, 2.3])
        
print ResultTable