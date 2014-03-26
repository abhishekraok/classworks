# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 16:33:40 2014

@author: Abhishek Rao

Title: XOR problem for Neural Network

Create 4 points corresponding to XOR problem for X and corresponding target vector y. 
Train the feed forward nerual network using backpropogation.
"""

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from itertools import izip
from sklearn.metrics import mean_squared_error

def scoreNN(X, y, net):
    """Evaluates the performance of NN by comparing the target vector y with predicted value using, i.e. net(X)"""
    return mean_squared_error(y, [net.activate(i)[0] for i in X])    

# Build a Neural Network with 2 input, 2 hidden layer and 1 output    
net = buildNetwork(2, 2, 1,) 
X = [(0,0), (1,1), (1,0), (0,1)] #XOR dataset
y = [0, 0, 1, 1]
ds = SupervisedDataSet(2, 1)
for i,j in izip(X,y):
    ds.addSample(i,j)
trainer = BackpropTrainer(net, dataset=ds,momentum = 0.9)

# Train and evaluate performance
score = []
for i in range(0,9999): # Number of epochs to train
    trainer.train()
    e = scoreNN(X,y,net)
    score.append(e)
    if e < 1e-3:
        break

# Results
figure(1)
plot(score)
title('Neural Network XOR problem')
xlabel('Iteration')
ylabel('MSE')

print 'Testing for input data [(0,0), (1,1), (1,0), (0,1)]'
print [net.activate(i)[0] for i in X]
print 'The Weights learnt are '
print net.params