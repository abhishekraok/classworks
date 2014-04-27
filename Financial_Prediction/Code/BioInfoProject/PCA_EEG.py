# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 19:47:11 2014

@author: Abhishek
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pylab as pl
from itertools import cycle
import sys
if not sys.path.count(".."): sys.path.append("..")
from sklearn.svm import SVC
    
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
    
############## Main ######################################3
X = np.loadtxt('../data/EEG_EYE.txt', delimiter = ',')
y = X[:,-1]           #y is the last column of data set
X = X[:,:-1]          #X is all of the data except last column
xpca = PCA(n_components=2).fit_transform(X)
plot_2D(xpca,y,['c1','c2'])

msvc=SVC()
msvc=msvc.fit(X,y)