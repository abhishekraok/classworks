# -*- coding: utf-8 -*-
"""
Created on Wed Apr 02 19:59:02 2014

@author: Abhishek

Echo self.state Networks library in "plain" scientific Python, 
originally by Mantas LukoÅ¡eviÄ?ius 2012
http://minds.jacobs-university.de/mantas/code
Modified by Abhishek Rao, trying to convert into crude scikit form
https://sites.google.com/site/abhishekraokrishna/
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ESN:
    """Builds an Echo self.state network class
    
    For more details visit http://minds.jacobs-university.de/sites/default/files/uploads/papers/Echoself.statesTechRep.pdf
    
    Parameters
    ----------
    inSize : float, optional (default=1.0)
        Dimension of the input vector.

    outSize : float, optional (default=1.0)
        Dimension of the output vector.

    resSize : float, optional (default=1000)
        The number of nodes in the reservoir.

    a : float, optional (default=0.3)
        Leak rate of the neurons.
         
    initLen : float, optional (default=100)
        Number of steps for initialization of reservoir. 

    trainLen : float, optional (default=2000)
        Number of steps for training of reservoir. 
        
    """
    def __init__(self, inSize=1, outSize=1, resSize=1000, 
                 a=0.3, initLen=100, trainLen=2000):
        random.seed(42)
        self.Win = (random.rand(resSize,1+inSize)-0.5) * 1
        self.W = random.rand(resSize,resSize)-0.5
        # Option 1 - direct scaling (quick&dirty, reservoir-specific):
        #W *= 0.135 
        # Option 2 - normalizing and setting spectral radius (correct, slow):
        self.rhoW = max(abs(linalg.eig(self.W)[0]))
        self.W *= 1.25 / self.rhoW
        self.resSize = resSize 
        self.a = a
        self.initLen = initLen
        self.trainLen = trainLen
        
    def fit(self, X, y):
        """Fit the ESN model according to the given training data.
        Note: Here X is used instead of u[n] for input following scikit
        learn convention. For the internal nodes state is used instead of X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : {array-like, sparse matrix}, shape (n_samples, n_features)
            Target values (real numbers in regression)
        Returns
        -------
        self : object
            Returns the ESN class.        
        """
        
#        # Making sure the input is numpy array, if not convert
#        if not X.__class__ == <type 'numpy.ndarray'>:
#            X = np.array(X)
#        if not y.__class__ == <type 'numpy.ndarray'>':
#            y = np.array(y) 

        self.outSize = y.shape[1]
        if len(X.shape) > 1 : # Check if array or matrix
            self.inSize = X.shape[1]
            Yt = X[None,self.initLen+1:self.trainLen+1]
        else:
            self.inSize = 1
            Yt = X[None,self.initLen+1:self.trainLen+1]
        
        self.testLen = self.trainLen 
        # allocated memory for the design (collected self.states) matrix
        self.state = np.zeros((1+self.inSize+self.resSize,self.trainLen-self.initLen))
        # set the corresponding target matrix directly
       
        
        # run the reservoir with the data and collect X
        self.x = np.zeros((self.resSize,1))
        for t in range(self.trainLen):
            u = X[t]
            self.x = (1-self.a)*self.x + self.a*tanh( dot( self.Win, vstack((1,u)) ) + dot( self.W, self.x ) )
            if t >= self.initLen:
                self.state[:,t-self.initLen] = vstack((1,u,self.x))[:,0]
                
        # train the output
        reg = 1e-8  # regularization coefficient
        self.state_T = self.state.T
        self.Wout = dot( dot(Yt,self.state_T), linalg.inv( dot(self.state,self.state_T) + \
                    reg*eye(1+self.inSize+self.resSize) ) )
        return self
        
    def predict(self, X, testLen=None):
        """Perform regression on samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
        
        testLen : float, optional (default=None)
                  Length of prediction. By default the value set during 
                  initialization is used.

        Returns
        -------
        y_pred : {array-like, sparse matrix}, shape (n_samples, n_features)
        """
        self.trainLen = X.shape[0]
        if testLen == None:
            testLen = self.testLen
        Y = np.zeros((testLen, self.outSize))
        for t in range(self.testLen):
            u = X[t] ## this would be a predictive mode:
            self.x = (1-self.a)*self.x + self.a*tanh( dot( self.Win, vstack((1,u)) ) + dot( self.W, self.x ) )
            y = np.dot( self.Wout, vstack((1,u,self.x)) )
            Y[t] = y
            # generative mode:
            # u = y
        return Y

############ Main #################
X = np.loadtxt('data/MackeyGlass_t17.txt')
y = X[100+1:2000+1, None] 
mod1 = ESN()
mod1 = mod1.fit(X,y)
yp = mod1.predict(X[2000:])
figure(1).clear()
plot( X, 'g' )
plot( yp, 'b' )
title('Target and generated signals $y(n)$ starting at $n=0$')
legend(['Target signal', 'Free-running predicted signal'])

figure(3).clear()
plot( range(len(mod1.Wout.T)), mod1.Wout.T)
title('Output weights $\mathbf{W}^{out}$')

show()
mse = sum( square( X[100+1:2000+1]  - y ) ) / y.shape[0]
print 'MSE = ' + str( mse )
############ End ###################
def originalcode():   
    # load the data
    trainLen = 2000
    testLen = 2000
    initLen = 100
    
    data = loadtxt('data/MackeyGlass_t17.txt')
    
    # plot some of it
    figure(10).clear()
    plot(data[0:1000])
    title('A sample of data')
    
    # generate the ESN reservoir
    inSize = outSize = 1
    resSize = 1000
    a = 0.3 # leaking rate
    
    random.seed(42)
    Win = (random.rand(resSize,1+inSize)-0.5) * 1
    W = random.rand(resSize,resSize)-0.5 
    # Option 1 - direct scaling (quick&dirty, reservoir-specific):
    #W *= 0.135 
    # Option 2 - normalizing and setting spectral radius (correct, slow):
    print 'Computing spectral radius...',
    rhoW = max(abs(linalg.eig(W)[0]))
    print 'done.'
    W *= 1.25 / rhoW
    
    # allocated memory for the design (collected self.states) matrix
    X = zeros((1+inSize+resSize,trainLen-initLen))
    # set the corresponding target matrix directly
    Yt = data[None,initLen+1:trainLen+1] 
    
    # run the reservoir with the data and collect X
    x = zeros((resSize,1))
    for t in range(trainLen):
        u = data[t]
        x = (1-a)*x + a*tanh( dot( Win, vstack((1,u)) ) + dot( W, x ) )
        if t >= initLen:
            X[:,t-initLen] = vstack((1,u,x))[:,0]
        
    # train the output
    reg = 1e-8  # regularization coefficient
    X_T = X.T
    Wout = dot( dot(Yt,X_T), linalg.inv( dot(X,X_T) + \
        reg*eye(1+inSize+resSize) ) )
    #Wout = dot( Yt, linalg.pinv(X) )
    
    # run the trained ESN in a generative mode. no need to initialize here, 
    # because x is initialized with training data and we continue from there.
    Y = zeros((outSize,testLen))
    u = data[trainLen]
    for t in range(testLen):
        x = (1-a)*x + a*tanh( dot( Win, vstack((1,u)) ) + dot( W, x ) )
        y = dot( Wout, vstack((1,u,x)) )
        Y[:,t] = y
        # generative mode:
        u = y
        ## this would be a predictive mode:
        #u = data[trainLen+t+1] 
    
    # compute MSE for the first errorLen time steps
    errorLen = 500
    mse = sum( square( data[trainLen+1:trainLen+errorLen+1] - Y[0,0:errorLen] ) ) / errorLen
    print 'MSE = ' + str( mse )
        
    # plot some signals
    figure(1).clear()
    plot( data[trainLen+1:trainLen+testLen+1], 'g' )
    plot( Y.T, 'b' )
    title('Target and generated signals $y(n)$ starting at $n=0$')
    legend(['Target signal', 'Free-running predicted signal'])
    
    figure(2).clear()
    plot( X[0:20,0:200].T )
    title('Some reservoir activations $\mathbf{x}(n)$')
    
    figure(3).clear()
    bar( range(1+inSize+resSize), Wout.T )
    title('Output weights $\mathbf{W}^{out}$')
    
    show()