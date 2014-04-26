# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 15:16:29 2014

@author: Abhishek
"""

import rlspy
import numpy as np
from numpy import linalg as LA

N_iter = 50

W = np.random.rand(2,4)
sm = 1e-2*np.random.rand(4,4)
A = [np.random.rand(4,2) for i in range(N_iter)]
yk = np.array([np.dot(a, W) + np.random.normal(0, sm) for a in A])

West = np.random.rand(2,4)
Pest = np.eye(4)
rlsi = rlspy.data_matrix.Estimator(West,Pest)

err =[]
for a,y in zip(A,yk):
    rlsi.update(a,y,np.eye(2))
    err.append(LA.norm(rlsi.x-W))
    
plot(err)