# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 17:26:46 2014

@author: Abhishek

Unsupervised classification of Iris dataset using EM for different initilization.
"""
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from numpy import *
import numpy as np
import math
from sklearn.decomposition import PCA
import pylab as pl
from itertools import cycle
from scipy.stats import mstats
from EMfx import *    
################## Main #############################
iris = load_iris()
X = np.matrix(iris.data)
N,D = X.shape
K = 3
ErrorRatioList = []
for somechoice in range(10):
    sigma = [np.eye(4) for i in range(K)]
    index = range(0,N-1)
    np.random.shuffle(index)        
    mu = X[index[:K]] #Random choices   
    #predicted = classify(X,mu,sigma,Pi) # Random initial prediction
    #plot_2D(X_pca, predicted, ["c0", "c1", "c2"])
    #plt.title('Initial Prediction')
    LLL,mu,sigma, Pi = EM(X,mu,sigma,K)
    predicted = classify(X,mu,sigma,Pi)
    c1,c2,c3 = mstats.mode(predicted[:50])[0][0], mstats.mode(predicted[50:100])[0][0], mstats.mode(predicted[100:150])[0][0]
    newclasstarget = [c1]*50 +[c2]*50 + [c3]*50
    ErrorRatio = np.average([i!=j for i,j in izip(newclasstarget, predicted)])
    ErrorRatioList.append(ErrorRatio)
    #plot_2D(X_pca, predicted, ["c0", "c1", "c2"])
    #plt.title('Prediction after EM')
    print 'The Error ratio in classification is ' + str(ErrorRatio)
