# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:57:58 2014

@author: Abhishek

Title: Custom Function library for Pattern Classification
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import izip
from mpl_toolkits.mplot3d import Axes3D

def classify(DiscriminantFx, X, Theta, weights=np.ones(2) ):
    """ 
    Classifies X into 0,1..n by choosing the max index i such that 
    weight(i)*DiscriminantFx(X,Theta(i)) is maximum. 
    Returns in the form of [[X1,y1], [X2,y2]...]
    """
    Result = []  
    for x in X:
        discriminatValues = [j*DiscriminantFx(x,Theta[i][0],Theta[i][1]) for i,j in izip(range(len(Theta)), weights)]
        maxdiscriminatValues = max(discriminatValues)        
        classdecision = [i for i,j in enumerate(discriminatValues) if j==maxdiscriminatValues][0]
        Result.append([x,classdecision])
    return Result

def MahalanobisD(x,mean,cov):
    """ Returns - (X-mu)*inv(sigma)*(X-mu) """   
    return float(-(np.matrix(x)- np.matrix(mean))*np.linalg.linalg.inv(cov)*(np.matrix(x)-np.matrix(mean)).T)

def predict(X, Theta):
    """ Predict y from given X by Classifying based on Mahalanobis distance """
    return [i[1] for i in classify(MahalanobisD,X, Theta, np.ones(len(Theta)))] 

def score(X, y, Theta):
    """ Takes the data X, classifies it according to CFx and then gives the mean accuracy 
    by comparing with y. 
    """
    return np.mean([i == j for i,j in izip(y,predict(X,Theta)) ])
    
def genGausXy(mean,cov, SIZE):
    """ Generate data X and label y SIZE times for each of the mean, covariance """
    X = [x for i,j in izip(mean,cov) for x in np.random.multivariate_normal(i,j,SIZE) ]
    y = [j for k in range(0,len(mean)) for j in k*np.ones(SIZE)]
    return X,y