# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 11:58:48 2014

@author: Abhishek
Spline dataset classification
"""
import csv
import numpy as np
import os
if not sys.path.count('..'): sys.path.append('..')
import ESNFile as es
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import cycle

def readSpliceData(filename):
    """Reads the gene file and converts into integers. 
    Parameters
    ----------

    filename : string,
        File to read
        
    Returns
    -------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vectors, where n_samples is the number of samples
        and n_features is the number of features. Integers 0,1..N where N
        is the alphabet size of the input.

    y : array-like, shape (n_samples,)
        Target values (integers) Integers 0,1..N where N
        is the alphabet size of the input.
    """
    X,y = [],[]
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
           y.append(row[0].strip())
           X.append(row[2].strip())
    # convert to integers
    Output_Alphabet = list(set(y))
    y = np.array([Output_Alphabet.index(i) for i in y])
    Input_Alphabet= ['A', 'C', 'T', 'G','D','N','S','R']
    X = np.array([[Input_Alphabet.index(j) for j in i] for i in X])
    return X,y

def plot_2D(data, target, target_names):
    """ Function to plot 2D data, taken from Scipy tutorial"""    
    colors = cycle('rgbcmykw')
    target_ids = range(len(target_names))
    plt.figure()
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(data[target == i, 0], data[target == i, 1],
                   c=c, label=label)
    plt.legend()
    
######################## Main ##########################    
plt.close('all')
X,y = readSpliceData('..\Data\splice.data')

#Machine learning part begins
Xtrain,Xtest, ytrain,ytest = train_test_split(X,y)
yp = SVC().fit(Xtrain,ytrain).predict(Xtest)
Accuracy = np.average([i==j for i,j in zip(yp,ytest)])

# Results
print 'The Accuracy using SVC is ', Accuracy
#---- No point in plotting time series data
#plt.figure()
#plt.plot(yp,'o',label='Predicted')
#plt.plot(ytest+0.01,'d',label='Actual')
#plt.axis([0,ytest.shape[0],-.1,2.1])
#plt.legend()
xpca = PCA(n_components=2).fit_transform(Xtest)
plot_2D(xpca,yp,Output_Alphabet)
plt.title('Predicted')
plot_2D(xpca,ytest,Output_Alphabet)
plt.title('Actual')