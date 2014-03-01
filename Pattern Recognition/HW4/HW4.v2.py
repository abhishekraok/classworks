# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:57:58 2014

@author: Abhishek

Problem 8, Computer excercise in Chapter 3 book Duda, Hart & Stork, Pattern Classification, 2ed
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import izip
from mpl_toolkits.mplot3d import Axes3D
import ..\CustomDisc as cd

def CovTest(dataset, emeani, CovSet, alpha, Title = 'Error'):
    """ Measures classification performance of set of covariance matrix CovSet 
    [[cov1,cov2,...covC], [cov1,cov2,...covC], ...]
    given dataset (X,y) and fixed single set of means emeani of the form [mean1, mean2,...]
    alpha is a parameter to display in the xaxis, with Title as Title.
    Plots it and returns the Score for each of the CovSet
    """ 
    X,y = dataset
    Score = []
    for s in ShrCovi:
        Theta = [[i,j] for i,j in izip(emeani,s)]
        Score += [cd.score(X,y,Theta)] 
    plt.figure()
    plt.plot(alpha,Score)
    plt.xlabel(' alpha ')
    plt.ylabel(' Ratio of correctly classified samples')
    plt.axis([0, 1, 0.5, 1])
    plt.title( Title +' (alpha)')
    plt.show()
    return Score
    
######################## MAIN ##############################################
#Consants
alpha = np.linspace(0,1,20)
SIZE = 20
# Generate training data
mean = [[0,0,0], [1,5,-3], [0,0,0]]
cov = [np.diag([3,5,2]), np.array([[1,0,0],[0,4,1],[0,1,6]]), 10*np.eye(3)]
X,y = cd.genGausXy(mean, cov, SIZE)
#Estimation of parameters
emeani = [np.mean(X[i*SIZE:(i+1)*SIZE], axis=0 ) for i in range(0,3)]
emean = np.mean(X, axis=0)
ecovi = [np.cov(X[i*SIZE:(i+1)*SIZE], rowvar=0 ) for i in range(0,3)]
ecov = np.cov(X, rowvar=0)
# Calculate Shrinkage
ni = SIZE
n = 3*ni
# part c
ShrCovi = [[ ((1-a)*ni*c + a*n*ecov)/((1-a)*ni + a*n) for c in ecovi] for a in alpha]
# part d, Classify for each shrinkage for training
ShrinkageTest((X,y),emeani, ShrCovi, alpha, 'Training Error')
# part e, Test Error, generate 50 new datapoints for each class 
ShrinkageTest(cd.genGausXy(mean, cov, 50), emeani, ShrCovi, alpha, 'Test Error')