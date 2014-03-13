# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 16:33:40 2014

@author: Abhishek

XOR problem for NN
"""

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from itertools import izip
from sklearn.metrics import mean_squared_error


def scoreNN(X, y, net):
    """Evaluates the score of NN by comparing y with net(X)"""
    return mean_squared_error(y, [net.activate(i)[0] for i in X])    
    
net = buildNetwork(2, 2, 1,)
X = [(0,0), (1,1), (1,0), (0,1)]
y = [0, 0, 1, 1]
ds = SupervisedDataSet(2, 1)
for i,j in izip(X,y):
    ds.addSample(i,j)
trainer = BackpropTrainer(net, dataset=ds,momentum = 0.9)

score = []
for i in range(0,9999):
    trainer.train()
    e = scoreNN(X,y,net)
    score.append(e)
    if e < 1e-3:
        break

figure(1)
plot(score)
title('Neural Network XOR problem')
xlabel('Iteration')
ylabel('MSE')
