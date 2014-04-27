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

def plot_2D(data, target, target_names):
    """ Function to plot 2D data
    Function taken from Scipy tutorial
    """    
    colors = cycle('rgbcmykw')
    target_ids = range(len(target_names))
    pl.figure()
    for i, c, label in zip(target_ids, colors, target_names):
        pl.scatter(data[target == i, 0], data[target == i, 1],
                   c=c, label=label)
    pl.legend()
    
# Data Preparation
X,y = [],[]
with open('..\Data\splice.data', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
       y.append(row[0].strip())
       X.append(row[2].strip())
# convert to integers
Output_Alphabet = list(set(y))
y = np.array([Output_Alphabet.index(i) for i in y])
Input_Alphabet= ['A', 'C', 'T', 'G','D','N','S','R']
X = np.array([[Input_Alphabet.index(j) for j in i] for i in X])

# Visualization
xpca = PCA(n_components=2).fit_transform(X)
plot_2D(xpca,y,Output_Alphabet)

#Machine learning part begins
Xtrain,Xtest, ytrain,ytest = train_test_split(X,y)
yp = SVC().fit(Xtrain,ytrain).predict(Xtest)
Accuracy = np.average([i==j for i,j in zip(yp,ytest)])
print 'The Accuracy is ', Accuracy
plt.figure()
plt.plot(yp,label='Predicted')
plt.plot(ytest,label='Actual')
plt.legend()