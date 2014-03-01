# -*- coding: utf-8 -*-
"""
Created on Sat Feb 08 15:40:57 2014

@author: Abhishek
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip
from mpl_toolkits.mplot3d import Axes3D
import CustomDisc as cd


mean1, mean2 = [1,-1], [-1, 1]
cov = np.matrix([ [1.01,0.2], [0.2,1.01] ])
Theta = [[mean1, cov], [mean2,cov]]
N = 1000
Partition = np.random.binomial(N, 2.0/3)
x1,y1 = np.random.multivariate_normal(mean1,cov,Partition).T
x2,y2 = np.random.multivariate_normal(mean2,cov, N-Partition).T
plt.plot(x1,y1,'x',label='Class 1')
plt.plot(x2,y2,'ro',label='Class 2'); 
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.axis('equal'); 
legend()
plt.show()

X1 = np.array([[x,y] for x,y in izip(x1,y1)])
X2 = np.array([[x,y] for x,y in izip(x2,y2)])
#Classify first class data
C1 = cd.classify(cd.MahalanobisD,X1, Theta)
Z1=[x[1] for x in C1]
C2 = cd.classify(cd.MahalanobisD,X2, Theta)
Z2=[x[1] for x in C2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1,y1,Z1)
ax.scatter(x2,y2,Z2,c='r',marker='v')

#Second part of the question, finding ROC
ROCx,ROCy = [],[]
weights = sorted(np.random.exponential(5,10))
for weight2 in weights:
    C1 = cd.classify(cd.MahalanobisD,X1, Theta,[1,weight2])
    Z1=[x[1] for x in C1]
    C2 = cd.classify(cd.MahalanobisD,X2, Theta,[1,weight2])
    Z2=[x[1] for x in C2]
    TPR = np.average(Z1)
    FPR = np.average(Z2)
    ROCx.append((FPR))
    ROCy.append((TPR))

plt.figure(5)
plt.plot(ROCx,ROCy,'o-')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()