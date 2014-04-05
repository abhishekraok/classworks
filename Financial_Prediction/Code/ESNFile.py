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

    """
    def __init__(self, inSize=1, outSize=1, resSize=1000, 
                 a=0.3, initLen=100, trainLen=2000):
        np.random.seed(42)
        self.Win = (np.random.rand(resSize,1+inSize)-0.5) * 1
        self.W = np.random.rand(resSize,resSize)-0.5
        # Option 1 - direct scaling (quick&dirty, reservoir-specific):
        #W *= 0.135 
        # Option 2 - normalizing and setting spectral radius (correct, slow):
        self.rhoW = max(abs(np.linalg.eig(self.W)[0]))
        self.W *= 1.25 / self.rhoW
        self.resSize = resSize 
        self.a = a
        self.initLen = initLen
        
    def fit(self, X, y, trainLen=None):
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
            
        trainLen : float, optional (default=None)
            Number of steps for training of reservoir. Default is the 
            length of X. i.e. X.shape[0]
                   
        Returns
        -------
        self : object
            Returns the ESN class.        
        """
        self.outSize = y.shape[1]
        if trainLen == None:
            trainLen = X.shape[0]
        if len(X.shape) > 1 : # Check if array or matrix
            self.inSize = X.shape[1]
            trainLen = min(trainLen, X.shape[0])
        else:
            self.inSize = 1
            trainLen = min(trainLen, len(X))
         # allocated memory for the design (collected self.states) matrix
        self.state = np.zeros((1+self.inSize+self.resSize,trainLen-self.initLen))
        # run the reservoir with the data and collect X
        self.x = np.zeros((self.resSize,1))
        for t in range(trainLen):
            u = X[t]
            self.x = (1-self.a)*self.x + self.a*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) )
            if t >= self.initLen:
                self.state[:,t-self.initLen] = np.vstack((1,u,self.x))[:,0]
                
        # train the output
        reg = 1e-8  # regularization coefficient
        self.state_T = self.state.T
        self.Wout = np.dot( np.dot(y[self.initLen:].T,self.state_T), np.linalg.inv( np.dot(self.state,self.state_T) + \
                    reg*np.eye(1+self.inSize+self.resSize) ) )
        return self
        
    def predict(self, X, testLen=None):
        """Perform regression on samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
        
        testLen : float, optional (default=None)
            Number of steps for prediction using the reservoir. Default 
            is the length of the input X.

        Returns
        -------
        y_pred : {array-like, sparse matrix}, shape (n_samples, n_features)
        
        """
        if testLen == None:
            testLen = X.shape[0]
        Y = np.zeros((testLen, self.outSize))
        
        for t in range( min(testLen,X.shape[0]) ):
            u = X[t] ## this would be a predictive mode:
            self.x = (1-self.a)*self.x + self.a*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) )
            y = np.dot( self.Wout, np.vstack((1,u,self.x)) )
            Y[t] = y
            # generative mode:
            # u = y
        return Y