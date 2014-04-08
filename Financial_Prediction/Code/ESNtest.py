# -*- coding: utf-8 -*-
"""
Created on Sat Apr 05 14:14:21 2014

@author: Abhishek
"""
import numpy as np
import matplotlib.pyplot as plt
import ESNFile as es

X = np.loadtxt('data/MackeyGlass_t17.txt')
#y = X[100+1:2000+1,None]     
#mod1 = es.ESN()
#mod1 = mod1.fit(X,y,trainLen=2000)
#yp = mod1.predict(X[2000:])
#figure(1).clear()
#plot( X[:2000], 'g' )
#plot( yp, 'b' )
#title('Target and generated signals $y(n)$ starting at $n=0$')
#legend(['Target signal', 'Free-running predicted signal'])
#
#figure(3).clear()
#plot( range(len(mod1.Wout.T)), mod1.Wout.T)
#title('Output weights $\mathbf{W}^{out}$')
#
#show()
#mse = sum( square( X[100+1:2000+1]  - y ) ) / y.shape[0]
#print 'MSE = ' + str( mse )


#X = np.sin(np.arange(0,50,0.01)) + np.sin(8*np.arange(0,50,0.01)) + np.sin(17*np.arange(0,50,0.01)) 
#X += 0.5*np.random.normal(size=X.shape)
#for i in [2,20,100,500]:
mod1 = es.ESN(resSize=9)
NMSE,yp,ytest = mod1.checkts(X)
plt.figure()
plt.plot(yp[:999],'o-',label='Predicted')
plt.plot(ytest[:999],'d--',label='Test target')
plt.legend()
plt.title('Prediction for reservoir size {0}'.format(i))
print 'For reservoir size {0} the NMSE is {1}'.format(i,NMSE) 
    
