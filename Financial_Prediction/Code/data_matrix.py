#!/usr/bin/env python
"""
Data matrix form for recursive least squares estimation
=======================================================
Originally from https://github.com/bluesquall/rlspy
Modified by Abhishek Rao https://github.com/abhishekraok/
Notations and details from 
Recursive Least Squares Estimation,(Com 477/577 Notes)
Yan-Bin Jia, Dec 12, 2013
https://www.cs.iastate.edu/~cs577/handouts/recursive-least-squares.pdf
"""

#TODO switch to 2-space indentation?
import itertools

import numpy as np

def priorgain(P, H, Rk):
    """Innovation gain calculated from a-priori parameters.

    Parameters
    ----------
    P : ndarray
        A priori estimation error covariance.
    H : ndarray
        Data coupling matrix
    Rk : ndarray
        Measurement noise covariance.

    note :: This is the Kalman gain calculated in the standard Kalman filter.

    """
    PHT = np.dot(P, H.T)
    Sk = np.dot(H, PHT) + Rk
    return np.linalg.solve(Sk.T, PHT.T).T

#def single_priorgain(P, A, v):
#    """Innovation gain calculated from a-priori parameters (for single point)
#
#    Parameters
#    ----------
#    P : ndarray
#        A priori estimation error covariance.
#    A : ndarray
#        Data coupling vector
#    v : float
#        Measurement noise variance.
#
#    note :: This is the Kalman gain calculated in the standard Kalman filter.
#
#    """
#    PAT = np.dot(P, A.T) # should be a column vector
#    APATpRk = np.dot(A, PAT) + v # should be a scalar
#    return PAT / APATpRk


#def posteriorgain(P, A, Rk):
#    """Innovation gain calculated from a-posteriori parameters.
#
#    Parameters
#    ----------
#    P : ndarray
#        A-posteriori estimation error covariance.
#    A : ndarray
#        Data coupling matrix
#    Rk : ndarray
#        Measurement noise covariance.
#
#    note :: This is the Kalman gain calculated using Swerling's method.
#
#    """
#    return reduce(np.dot, P, A.T, np.linalg.inv(Rk))
#    # TODO possibly change to use inv(Rk) as arg 
#    #   ...(but we're not planningon using this much)
#

class Estimator(object):
    """Recursive Least Squares estimator (data matrix form)

    """

    def __init__(self, x0, P0):
        """Returns new RLS estimator."""
        self.x = x0
        self.P = P0
        # identity matrix same size as P for convenience
        self.I = np.identity(len(x0)) 


    def __call__(self, A, b, Rk):
        """(convenience method) run self.update with given inputs."""
        self.update(A, b, Rk)


    def update(self, H, yk, Rk):
        """Recursively update the estimate given new data.

        TODO: longer explanation with math

        Parameters
        ----------
        H : ndarray
            Data coupling matrix. (m-by-n)
        yk : ndarray
            Data vector. (n-by-1)

        """
        K = priorgain(self.P, H, Rk) # Kalman Gain
        y = yk - np.dot(H, self.x) # innovation
        self.x = self.x + np.dot(K, y) # a posteriori estimate
        self.P = np.dot(self.I - np.dot(K, H), self.P) # a posteriori covariance
#
#    def single_update(self, A, b, v):
#        """Recursively update given a single point of new data.
#
#        TODO: motivation, longer explanation with math
#        motivation: gets rid of np.solve (potentially costly)
#
#        Parameters
#        ----------
#        A : ndarray
#            Data coupling matrix. (m-by-1)
#        b : float
#            Data point. (1-by-1)
#        v : float
#            Measurement variance.
#
#        Returns
#        -------
#        nothing (self.x and self.P are updated)
#
#        """
##        A = A.reshape(1, -1) # TODO consider np.atleast2d
#        K = single_priorgain(self.P, A, v)
#        y = b - np.dot(A, self.x)
#        self.x = self.x + K * y # a posteriori estimate
#        self.P = np.dot(self.I - np.dot(K, A), self.P) # a posteriori covariance
#
#
#    def consecutive_update(self, A, b, v):
#        """Use consecutive single-row updates given several new rows of data.
#
#        TODO: motivation, longer explanation with math
#
#        Parameters
#        ----------
#        A : ndarray
#            Data coupling matrix. (m-by-1)
#        b : ndarray
#            Data vector. (n-by-1)
#        v : float
#            Measurement variance.
#
#        Returns
#        -------
#        nothing (self.x and self.P are updated)
#
#        """
#        # for Ar, br in zip(A, b): self.single_update(Ar, br, v)
#        [self.single_update(Ar.reshape(1, -1), br, v) 
#            for Ar, br in itertools.izip(A, b)]
#
#
#
## TODO additional Estimator classes with alternate updates 
##       (e.g., Swerling posterior, square root forms)
