# -*- coding: utf-8 -*-
"""
Created on Wed Apr 02 19:59:02 2014

@author: Abhishek Rao

Echo self.state Networks library in "plain" scientific Python, 
originally by Mantas Lukosevicius 2012
http://minds.jacobs-university.de/mantas/code
Modified by Abhishek Rao, trying to convert into crude scikit form
https://sites.google.com/site/abhishekraokrishna/
"""
import numpy as np
import data_matrix as rls
from sklearn.svm import SVC
from sklearn.base import RegressorMixin, BaseEstimator

def ts_train_test_split(X, test_size=0.3, lags=0):
    """ Converts time series data into Xtrain, Xtest, Ytrain and Ytest.
    
    Y is calculated from X using Y[n] = x[n+1]. Therefore 
    y = np.roll(X). X converted from a (n_samples,n_features) matrix into 
    (n_samples,n_features x (lags+1)) matrix.
       
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Time Series vectors, where n_samples is the number of samples
            and n_features is the number of features.
            
        test_size : float, optional (default=0.3)
            Ratio of test length to train length returned.
            
        lags : float, optional (default=0)
            The number of previous data values to pack in each X.
            e.g. if lags = P then X'[n] = [X[n], X[n-1], ..X[n-P]]
                   
        Returns
        -------
        Xtrain : {array-like, sparse matrix}, shape (n_samples x (1-testRatio),
                                                        n_features x (lags+1))
            Train input values.
            
        Xtest : {array-like, sparse matrix}, shape (n_samples x (testRatio), 
                                                        n_features x (lags+1))
            Test input values.

        Ytrain : {array-like, sparse matrix}, shape (n_samples x (1-testRatio), 
                                                        n_features x (lags+1))
            Target values for training.

        Ytest : {array-like, sparse matrix}, shape (n_samples x (testRatio), 
                                                        n_features x (lags+1))
            Target test values.
            
        """   
        
    Y = X[1 + lags:]
    if lags > 0:
        X = np.array([X[i:i+lags+1].flatten() for i in range(len(X) - 
                                                                    lags - 1)])
    else:
        X = X[:-1]
    Ntrain = int(np.round( len(X)*(1 - test_size) ))
    return X[:Ntrain], X[Ntrain:], Y[:Ntrain], Y[Ntrain:]


class ESN(RegressorMixin,BaseEstimator):
    """Builds an Echo self.state network class
    
    For more details visit http://minds.jacobs-university.de/sites/default/
                                files/uploads/papers/Echoself.statesTechRep.pdf
    
    Parameters
    ----------
    resSize : float, optional (default=1000)
        The number of nodes in the reservoir.

    leakRate : float, optional (default=0.3)
        Leak rate of the neurons.
         
    initLen : float, optional (default=0)
        Number of steps for initialization of reservoir. If no option
        passed then length of input divided by 10 is used.

    """
    def __init__(self, resSize=100, leakRate = 0.3, initLen=0):
        np.random.seed(42)
        self.W = np.random.rand(resSize,resSize)-0.5
        self.rhoW = max(abs(np.linalg.eig(self.W)[0]))
        self.W *= 1.25 / self.rhoW
        self.resSize = resSize 
        self.leakRate = leakRate
        self.initLen = initLen
    
    def setparams(self,X,y=None, trainLen=None):
        """ Sets parameters based on input X,y"""
        if y == None:
            y = np.roll(X, shift =-1, axis=0)
        self.outSize = y.shape[1] if len(y.shape) > 1 else 1
        self.actual_output_size = y.shape[1] if len(y.shape) > 1 else None
        if trainLen == None:
            self.trainLen = X.shape[0]
        if len(X.shape) > 1 : # Check if array or matrix
            self.inSize = X.shape[1]
            self.trainLen = min(self.trainLen, X.shape[0])
        else:
            self.inSize = 1
            self.trainLen = min(self.trainLen, len(X))
        self.Win = (np.random.rand(self.resSize, 1 + self.inSize) - 0.5) * 1
         # allocated memory for the design (collected self.states) matrix
        self.Nfin = 1 + self.resSize + self.inSize
        self.state = np.zeros((self.Nfin, self.trainLen-self.initLen))
        return self
        
    def runreservoir(self, X):
        """ Run reservoir with data X and generate Xstates"""
        self.x = np.zeros((self.resSize, 1))
        Xstate = np.zeros((self.Nfin, X.shape[0]-self.initLen))
        for t in range(X.shape[0]):
            u = X[None, t].T
            self.x = (1-self.leakRate)*self.x + self.leakRate*np.tanh(
                np.dot( self.Win, np.vstack((1,u))) + np.dot( self.W, self.x ))
            if t >= self.initLen:
                Xstate[:, t-self.initLen] = np.vstack((1, u, self.x))[:, 0]
        return Xstate
        
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
        self = self.setparams(X, y, trainLen) # Set parameters, initialization
        self.state = self.runreservoir(X)   # Run the reservoir for input X
        # train the output
        reg = 1e-8  # regularization coefficient
        self.state_T = self.state.T
        self.Wout = np.dot(np.dot(y[self.initLen:].T, self.state_T), 
                           np.linalg.inv(np.dot(self.state, self.state_T) + \
                           reg*np.eye(self.Nfin)))
        self.alphabet = list(set(y))
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
        Y = np.zeros((testLen,self.outSize if not self.outSize == None else 1))
        
        for t in range(min(testLen, X.shape[0])):
            u = X[None, t].T # this would be a predictive mode:
            self.x = (1 - self.leakRate)*self.x + self.leakRate*np.tanh(
                np.dot( self.Win, np.vstack((1,u))) + np.dot(self.W, self.x ))
            y = np.dot(self.Wout, np.vstack((1, u, self.x)))
            Y[t] = y.reshape(Y[0].shape)
        if self.actual_output_size == None:
            Y = Y.reshape(-1)
        return Y
        
    def checkts(self,X):
        """ Tests timeseries X's performance in ESN.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------
        NMSE : {float}
               The normalized Mean Square Error of predicted output (yp)
               and actual output (ytest)
               
        yp : {array-like, sparse matrix}, shape (n_samples, n_features)       
             The predicted value of output based on Xtest. 
             yp = ESN.predict(Xtest)
             
        ytest : {array-like, sparse matrix}, shape (n_samples, n_features)
               The actual target values in the test set.
               
        """
        
        Xtrain,Xtest,ytrain,ytest = ts_train_test_split(X)
        self = self.fit(Xtrain, ytrain)
        return self.score(Xtest,ytest), self.predict(Xtest), ytest
   
    def adaptfitpredict(self, X, y):
        """Fit the ESN model adaptively and predict according to 
        the given training data. 
        Notations: Here X is used instead of u[n] for input following scikit
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
        y_pred : {array-like, sparse matrix}, shape (n_samples, n_features)
            Predicted value of target(n+1) ~= y(n) = Wout*[x(n) u(n)]
        """
        
        # Initializtions
        self = self.setparams(X, y)
        Y = np.zeros((self.trainLen-self.initLen, self.outSize))
        self.Nfin = self.resSize+self.inSize +1
        
        # Recursive Least Square solution for Wout
        self.Wout = np.random.rand(self.outSize, self.Nfin)-0.5
        P0i = [np.identity(self.Nfin) for i in range(self.outSize)]
        RLSWout = [rls.Estimator(self.Wout[i, :].reshape(-1, 1),P0i[i]) for 
                                                    i in range(self.outSize)]    
        Rk = 1e-2 # This is not necessary:all our measurements have equal var
        
        # run the reservoir with the data and collect X
        self.x = np.zeros((self.resSize,1))
        for t in range(self.trainLen):
            u = X[None, t].T
            self.x = (1-self.leakRate)*self.x + \
                   self.leakRate*np.tanh(np.dot(self.Win, np.vstack((1,u))) + \
                   np.dot( self.W, self.x ) )
            yp = np.dot( self.Wout, np.vstack((1, u, self.x))) # predict
            if t >= self.initLen:  # write results only after transients
                self.state[:, t-self.initLen] = np.vstack((1, u, self.x))[:, 0]
                Y[t-self.initLen] = yp.reshape(Y[0].shape)
            # Adapt Wout
            for k, (pmat, rlsi) in enumerate(zip(P0i, RLSWout)):
                yact = y[t, k].reshape(-1, 1) if len(y.shape) > 1 else y[t]
                rlsi.update(np.vstack((1, u, self.x)).reshape(1, -1), yact, Rk)
                self.Wout[k] = rlsi.x.T
        Y = Y.reshape(y[self.initLen:].shape)
        return Y


class ESNHardLimiter(ESN):
    """Builds an Echo State Network with hard limiter to adjust output
    
    Subclass of ESN; Changes predict method
    
    Parameters
    ----------
    resSize : float, optional (default=1000)
        The number of nodes in the reservoir.

    a : float, optional (default=0.3)
        Leak rate of the neurons.
         
    initLen : float, optional (default=0)
        Number of steps for initialization of reservoir. If no option
        passed then length of input divided by 10 is used.

    """
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
        y: {array-like, sparse matrix}, shape (n_samples, n_features)
            Class labels for samples in X
        """
        y = ESN.predict(self, X, testLen)
        # convert y predictions to closest integer in alphabet (0, 1, or 2)
        y = np.array([self.alphabet[np.argmin(
                [abs(yi - i) for i in self.alphabet])] for yi in y])
        
        if self.outSize == None:
            y = y.reshape(y.shape[0])
        return y

class ESNC(ESN):
    """Builds an Echo State Network Classifier Class
    
    Subclass of ESN; Modifies fit, predict, and adaptfitpredict
    
    Parameters
    ----------
    resSize : float, optional (default=1000)
        The number of nodes in the reservoir.

    a : float, optional (default=0.3)
        Leak rate of the neurons.
         
    initLen : float, optional (default=None)
        Number of steps for initialization of reservoir. If no option
        passed then length of input divided by 10 is used.

    """
    
    def fit(self, X, y, trainLen=None):
        """Fit the ESNC model according to the given training data.
        Note: Here X is used instead of u[n] for input following scikit
        learn convention. For the internal nodes state is used instead of X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : {array-like, sparse matrix}, shape (n_samples, n_features)
            Target values (binary)
            
        trainLen : float, optional (default=None)
            Number of steps for training of reservoir. Default is the 
            length of X. i.e. X.shape[0]
                   
        Returns
        -------
        self : object
            Returns the ESN class.        
        """
        self = self.setparams(X, y, trainLen)
        self.state = self.runreservoir(X)        
        y = y[self.initLen:]                    # adjust size accd. to initLen
        self.svmod = SVC().fit(self.state.T, y)  # fit SVC model
        return self
        
    def predict(self, X, testLen=None):
        """Perform classification on samples in X using ESN Classifier.

        For a one-class model, +1 or -1 is returned. Uses SVM of sklearn.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
        
        testLen : float, optional (default=None)
            Number of steps for prediction using the reservoir. Default 
            is the length of the input X.

        Returns
        -------
        y: {array-like, sparse matrix}, shape (n_samples, n_features)
            Class labels for samples in X.
        
        """
        return self.svmod.predict(self.runreservoir(X).T)                       

    

class ESNAdapt(ESN):
    """Builds an Echo State Network with adaptive Prediction.
    
    Subclass of ESN; Fit method does nothing. Changes Predict method.
    
    Parameters
    ----------
    resSize : float, optional (default=1000)
        The number of nodes in the reservoir.

    a : float, optional (default=0.3)
        Leak rate of the neurons.
         
    initLen : float, optional (default=0)
        Number of steps for initialization of reservoir. If no option
        passed then length of input divided by 10 is used.

    """
    def __init__(self, resSize=100, leakRate = 0.3, initLen=0):
        super(ESNAdapt, self).__init__(
        resSize, leakRate, initLen)
        
    def fit(self, X, y):
        """ Does nothing in Adaptive class. Put here to prevent 
        unneccesary computation"""
        return self
    
    def predict(self, X, y=None):
        """Fit the ESN model adaptively and predict according to 
        the given training data. 
        Notations: Here X is used instead of u[n] for input following scikit
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
        y_pred : {array-like, sparse matrix}, shape (n_samples, n_features)
            Predicted value of target(n+1) ~= y(n) = Wout*[x(n) u(n)]
        """
        
        # Initializtions
        self = self.setparams(X, y)
        if y == None:
            y = np.roll(X, shift =-1, axis=0)
        Y = np.zeros((self.trainLen-self.initLen, self.outSize))
        self.Nfin = self.resSize+self.inSize +1
        
        # Recursive Least Square solution for Wout
        self.Wout = np.random.rand(self.outSize, self.Nfin)-0.5
        P0i = [np.identity(self.Nfin) for i in range(self.outSize)]
        RLSWout = [rls.Estimator(self.Wout[i, :].reshape(-1, 1),P0i[i]) for 
                                                    i in range(self.outSize)]    
        Rk = 1e-2 # This is not necessary:all our measurements have equal var
        
        # run the reservoir with the data and collect X
        self.x = np.zeros((self.resSize,1))
        for t in range(self.trainLen):
            u = X[None, t].T
            self.x = (1-self.leakRate)*self.x + \
                   self.leakRate*np.tanh(np.dot(self.Win, np.vstack((1,u))) + \
                   np.dot( self.W, self.x ) )
            yp = np.dot( self.Wout, np.vstack((1, u, self.x))) # predict
            if t >= self.initLen:  # write results only after transients
                self.state[:, t-self.initLen] = np.vstack((1, u, self.x))[:, 0]
                Y[t-self.initLen] = yp.reshape(Y[0].shape)
            # Adapt Wout
            for k, (pmat, rlsi) in enumerate(zip(P0i, RLSWout)):
                yact = y[t, k].reshape(-1, 1) if len(y.shape) > 1 else y[t]
                rlsi.update(np.vstack((1, u, self.x)).reshape(1, -1), yact, Rk)
                self.Wout[k] = rlsi.x.T
        Y = Y.reshape(y[self.initLen:].shape)
        return Y