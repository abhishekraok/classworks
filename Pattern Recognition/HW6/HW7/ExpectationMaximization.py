# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 14:57:42 2014

@author: Abhishek

Unsupervised classification of Iris dataset using EM.
"""
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from numpy import *
import numpy as np
import math

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


################## Main #############################
iris = load_iris()
X = np.matrix(iris.data)
N,D = X.shape
K = 3 # Number of clusters
Pi = (1.0/K)*ones(K)
Nk = Pi
sigma = [np.eye(4) for i in range(K)]
new_sigma = sigma
Gamma = np.zeros((N,K)) # Responsibilites , posterior probabillity
LogLk = 0
new_LogLk = 0
Gammafx = lambda n,mu,sigma: sum([Pi[k]*norm_pdf_multivariate(X[n],mu[k],sigma[k]) for k in range(K)])
epsilon = 0.01 # Convergence limit

plt.figure()
for initialchoice in range(5):  # Different initial choice of starting means
    mu = [X[random.randint(0,N-1)] for i in range(K)] #Random choices
    new_mu = mu
    LLL = [] # Log likelihood list to plot
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
        LLL.append(new_LogLk)
        if abs(new_LogLk - LogLk) < epsilon:
            break
        else:
            mu, sigma, LogLk = new_mu, new_sigma, new_LogLk
    
    # Show Results
    
    plot(LLL,'o-',label='Initialization #{0}'.format(initialchoice))
plt.title('Log likelihood vs iterations')
plt.ylabel('Log likelihood')
plt.xlabel('Iteration number')
plt.legend(loc=4)