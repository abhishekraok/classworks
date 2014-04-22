# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 17:26:46 2014

@author: Abhishek

Unsupervised classification of Iris dataset using EM for different K.
"""
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from numpy import *
import numpy as np
import math
from EMfx import *

################## Main #############################
iris = load_iris()
X = np.matrix(iris.data)
N,D = X.shape
Klist = range(3,7) # Number of clusters
LLL = [] # Log likelihood list to plot

for K in Klist:  # Different Number of clusters
    sigma = [np.eye(4) for i in range(K)]
    index = range(0,N-1)
    np.random.shuffle(index)        
    mu = X[index[:K]] #Random choices       
    LLL.append(EM(X,mu,sigma,K)[0])   
# Show Results
BIC = [-2*LLL[k] + Klist[k]*(log(N)+log(2*pi)) for k in range(len(LLL))]
plt.figure()
#plt.axis([2.5,6.5])   
plot(Klist,LLL,'o--')
plt.title('Log likelihood vs Number of clusters K')
plt.ylabel('Log likelihood')
plt.xlabel('Number of clusters')
plt.figure()
plot(Klist,BIC,'o--')
plt.title('BIC vs Number of clusters K')
plt.ylabel('BIC')
plt.xlabel('Number of clusters')