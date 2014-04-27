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
import data_matrix as rls
from sklearn.svm import SVC

def ts_train_test_split(X, test_size=0.3, lags=0):
    """ Converts time series data into Xtrain, ytrain, Xtest and ytest.
    
    y is calculated from X using y[n]=x[n+1]. Therefore 
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
            e.g. if lags = P then X'[n] = [X[n],X[n-1],..X[n-P]]
                   
        Returns
        -------
        Xtrain : {array-like, sparse matrix}, shape (n_samples x (1-testRatio), n_features x (lags+1))
            Train input values.
            
        Xtest : {array-like, sparse matrix}, shape (n_samples x (testRatio), n_features x (lags+1))
            Test input values.

        ytrain : {array-like, sparse matrix}, shape (n_samples x (1-testRatio), n_features x (lags+1))
            Target values for training.

        ytest : {array-like, sparse matrix}, shape (n_samples x (testRatio), n_features x (lags+1))
            Target test values.
            
        """   
        
    y = X[1+lags:]
    if lags > 0:
        X = np.array([X[i:i+lags+1].flatten() for i in range(len(X)-lags-1)])
    else:
        X = X[:-1]
    Ntrain = int(np.round( len(X)*(1-test_size) ))
    return X[:Ntrain], X[Ntrain:], y[:Ntrain], y[Ntrain:]


class ESN:
    """Builds an Echo self.state network class
    
    For more details visit http://minds.jacobs-university.de/sites/default/files/uploads/papers/Echoself.statesTechRep.pdf
    
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
    def __init__(self, resSize=100, 
                 a=0.3, initLen=None):
        np.random.seed(42)
        self.W = np.random.rand(resSize,resSize)-0.5
        # Option 1 - direct scaling (quick&dirty, reservoir-specific):
        #W *= 0.135 
        # Option 2 - normalizing and setting spectral radius (correct, slow):
        self.rhoW = max(abs(np.linalg.eig(self.W)[0]))
        self.W *= 1.25 / self.rhoW
        self.resSize = resSize 
        self.a = a
        self.initLen = initLen
    
    def setparams(self,X,y, trainLen=None):
        """ Sets parameters based on input X,y"""
        self.outSize = y.shape[1] if len(y.shape) > 1 else 1
        if self.initLen == None:
            self.initLen = int(np.floor(len(X)/10))
        if trainLen == None:
            self.trainLen = X.shape[0]
        if len(X.shape) > 1 : # Check if array or matrix
            self.inSize = X.shape[1]
            self.trainLen = min(self.trainLen, X.shape[0])
        else:
            self.inSize = 1
            self.trainLen = min(self.trainLen, len(X))
        self.Win = (np.random.rand(self.resSize,1+self.inSize)-0.5) * 1
         # allocated memory for the design (collected self.states) matrix
        self.Nfin = 1 +self.resSize +self.inSize
        self.state = np.zeros((self.Nfin,self.trainLen-self.initLen))
        return self
        
    def runreservoir(self,X):
        """ Run reservoir with data X and generate Xstates"""
        self.x = np.zeros((self.resSize,1))
        Xstate = np.zeros((self.Nfin,X.shape[0]-self.initLen))
        for t in range(X.shape[0]):
            u = X[None,t].T
            self.x = (1-self.a)*self.x + self.a*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) )
            if t >= self.initLen:
                Xstate[:,t-self.initLen] = np.vstack((1,u,self.x))[:,0]
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
        self = self.setparams(X,y,trainLen) # Set parameters, initialization
        self.state = self.runreservoir(X)         # Run the reservoir for input X
        # train the output
        reg = 1e-8  # regularization coefficient
        self.state_T = self.state.T
        self.Wout = np.dot( np.dot(y[self.initLen:].T,self.state_T), np.linalg.inv( np.dot(self.state,self.state_T) + \
                    reg*np.eye(self.Nfin) ) )
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
        Y = np.zeros((testLen, self.outSize if not self.outSize == None else 1))
        
        for t in range( min(testLen,X.shape[0]) ):
            u = X[None,t].T ## this would be a predictive mode:
            self.x = (1-self.a)*self.x + self.a*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) )
            y = np.dot( self.Wout, np.vstack((1,u,self.x)) )
            Y[t] = y.reshape(Y[0].shape)
            # generative mode:
            # u = y
        if self.outSize == None:
            Y = Y.reshape(Y.shape[0])
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
        self = self.fit(Xtrain,ytrain)
        yp = self.predict(Xtest)
        return (yp-ytest).var() / ytest.var(), yp, ytest
        
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
        self = self.setparams(X,y)
        Y = np.zeros((self.trainLen-self.initLen, self.outSize))
        self.Nfin = self.resSize+self.inSize +1
        # End of initializations 
        
        # Recursive Least Square solution for Wout
        self.Wout = np.random.rand(self.outSize, self.Nfin)-0.5
        P0i = [np.identity(self.Nfin) for i in range(self.outSize)]
        RLSWout = [rls.Estimator(self.Wout[i,:].reshape(-1,1),P0i[i]) for i in range(self.outSize)]    
        Rk = 1e-2 # This is not really needed, all our measurements
        #.. have equal variance , not sure
        # Loop begins
        # run the reservoir with the data and collect X
        self.x = np.zeros((self.resSize,1))
        for t in range(self.trainLen):
            u = X[None,t].T
            self.x = (1-self.a)*self.x + self.a*np.tanh(np.dot(self.Win, np.vstack((1,u))) + np.dot( self.W, self.x ) )
            yp = np.dot( self.Wout, np.vstack((1,u,self.x))) # Predict
            if t >= self.initLen:  # Write results only after transients
                self.state[:,t-self.initLen] = np.vstack((1,u,self.x))[:,0]
                Y[t-self.initLen] = yp.reshape(Y[0].shape)
            # Adapt Wout
            for k,(pmat,rlsi) in enumerate(zip(P0i,RLSWout)):
                yact = y[t,k].reshape(-1,1) if len(y.shape) > 1 else y[t]
                rlsi.update(np.vstack((1,u,self.x)).reshape(1,-1),yact,Rk)
                self.Wout[k] = rlsi.x.T
        Y = Y.reshape(y[self.initLen:].shape)
        return Y

class ESNC(ESN):
    """Builds an Echo State Network Classifier Class
    
    Modified version of visit http://minds.jacobs-university.de/sites/default/files/uploads/papers/Echoself.statesTechRep.pdf
    
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
        self = self.setparams(X,y,trainLen) # Set parameters, initialization
        self.state = self.runreservoir(X)         # Run the reservoir for input X
        y = y[self.initLen:]
        self.svmod = SVC().fit(self.state.T,y)
        # train the outpu
        return self
        
    def predict(self, X, testLen=None):
        """Perform classification on samples in X usin ESN Classifier.

        For an one-class model, +1 or -1 is returned. Uses SVM of sklearn.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Class labels for samples in X.
        """
        return self.svmod.predict(self.runreservoir(X).T)

    def adaptfitpredict(self, X, y):
        """Fit the ESNC model adaptively and predict according to 
        the given training data. 
        Notations: Here X is used instead of u[n] for input following scikit
        learn convention. For the internal nodes state is used instead of X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : {array-like, sparse matrix}, shape (n_samples, n_features)
            Target values ({0,1} in classification)

                   
        Returns
        -------
        y_pred : {array-like, sparse matrix}, shape (n_samples, n_features)
            Predicted value of target(n+1) ~= y(n) = Wout*[x(n) u(n)]
            
        self : object
            Returns the ESN class.        

        """
        
        # Initializtions
        self.outSize = y.shape[1] if len(y.shape) > 1 else 1
        if self.initLen == None:
            self.initLen = int(np.floor(len(X)/10))
        self.trainLen = X.shape[0]
        if len(X.shape) > 1 : # Check if array or matrix
            self.inSize = X.shape[1]
            self.trainLen = min(self.trainLen, X.shape[0])
        else:
            self.inSize = 1
            self.trainLen = min(self.trainLen, len(X))
        self.Win = (np.random.rand(self.resSize,1+self.inSize)-0.5) * 1
         # allocated memory for the design (collected self.states) matrix
        self.state = np.zeros((self.Nfin,self.trainLen-self.initLen))
        Y = np.zeros((self.trainLen, self.outSize))
        self.Nfin = self.resSize+self.inSize +1
        # End of initializations 
        
        # Recursive Least Square solution for Wout
        
        self.Wout = np.random.rand(self.outSize, self.Nfin)-0.5
        P0i = [np.identity(self.Nfin) for i in range(self.outSize)]
        RLSWout = [rls.Estimator(self.Wout[i,:].reshape(-1,1),P0i[i]) for i in range(self.outSize)]    
        Rk = 1e-2 # This is not really needed, all our measurements
        #.. have equal variance
        # Loop begins
        # run the reservoir with the data and collect X
        self.x = np.zeros((self.resSize,1))
        for t in range(self.trainLen):
            u = X[None,t].T
            self.x = (1-self.a)*self.x + self.a*np.tanh(np.dot(self.Win, np.vstack((1,u))) + np.dot( self.W, self.x ) )
            if t >= self.initLen:
                self.state[:,t-self.initLen] = np.vstack((1,u,self.x))[:,0]
                yp = np.greater(np.dot( self.Wout, np.vstack((1,u,self.x))),0.5)
                for k,(pmat,rlsi) in enumerate(zip(P0i,RLSWout)):
                    yact = y[t,k].reshape(-1,1) if len(y.shape) > 1 else y[t]
                    rlsi.update(np.vstack((1,u,self.x)).reshape(1,-1),yact,Rk)
                    self.Wout[k] = rlsi.x.T
                Y[t] = yp.reshape(Y[0].shape)
        Y = Y.reshape(y.shape)
        #Debug code 
#        reg = 1e-8  # regularization coefficient
#        self.state_T = self.state.T
#        self.Wouti = np.dot( np.dot(y[self.initLen:].T,self.state_T), np.linalg.inv( np.dot(self.state,self.state_T) + \
#                    reg*np.eye(self.Nfin) ) )
        return Y