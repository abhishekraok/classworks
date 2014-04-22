# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 17:26:46 2014

@author: Abhishek Rao

EM functions.
"""

from numpy.linalg import inv
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

def classify(X,mu,sigma,Pi):
    """ Classifies the given X into cluster 0,1...K-1 depending on the highest value"""
    N,D = X.shape
    K = len(Pi)
    return np.array([argmax([Pi[k]*norm_pdf_multivariate(X[n],mu[k],sigma[k]) for k in range(K)]) for n in range(N)])

def norm_pdf_multivariate(x, mu, sigma):
  """ Evaluates the Gaussian normal pdf for given input of point, mean and covariance matrix.  
  Modified code, function obtained from http://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python"""
  size = x.shape[1]
  if size == mu.shape[1] and (size, size) == sigma.shape:
    det = linalg.det(sigma)
    if det == 0:
        raise NameError("The covariance matrix can't be singular")

    norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
    x_mu = matrix(x - mu)
    result = math.pow(math.e, -0.5 * (x_mu * inv(sigma) * x_mu.T))
    return norm_const * result
  else:
    raise NameError("The dimensions of the input don't match")

def EM(X,mu,sigma, K):
    """ Perform EM algorithm with given mean, covariance for given number 
    of clusters until log likelihood change is less than epsilon = 0.01
    """
    N,D = X.shape
    new_sigma = sigma
    new_mu = mu
    Pi = (1.0/K)*ones(K)
    Nk = Pi
    Gamma = np.zeros((N,K)) # Responsibilites , posterior probabillity
    LogLk = 0
    new_LogLk = 0
    Gammafx = lambda n,mu,sigma: sum([Pi[k]*norm_pdf_multivariate(X[n],mu[k],sigma[k]) for k in range(K)])
    epsilon = 0.01 # Convergence limit
    # EM Loop begins
    while True:
        # E step
        for n in range(N):
            sumGamma = Gammafx(n,mu,sigma)
            for k in range(K):
                Gamma[n][k] = Pi[k]*norm_pdf_multivariate(X[n],mu[k],sigma[k]) / sumGamma
        # M step
        for k in range(K):
            Nk[k] = sum([Gamma[n][k] for n in range(N)])
            new_mu[k] = (1/Nk[k])*sum([Gamma[n][k]*X[n] for n in range(N)], axis=0)
            new_sigma[k] = (1/Nk[k])*sum([Gamma[n][k]*(X[n]-mu[k]).T*(X[n]-mu[k]) for n in range(N)], axis=0)
        Pi = Nk/N
        # Check Log likelihood
        new_LogLk = sum( [log(Gammafx(n,new_mu,new_sigma)) for n in range(N)],axis=0)
        if abs(new_LogLk - LogLk) < epsilon:
            break
        else:
            mu, sigma, LogLk = new_mu, new_sigma, new_LogLk
    return new_LogLk, mu, sigma, Pi

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