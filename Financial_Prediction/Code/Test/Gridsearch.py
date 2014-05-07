# -*- coding: utf-8 -*-
"""
Created on Tue May 06 22:23:48 2014

@author: Abhishek
"""

import numpy as np
import pylab as pl
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import sys
if not sys.path.count(".."): sys.path.append("..")
import ESNFile as es
from sklearn.preprocessing import normalize

##############################################################################
# Load and prepare data set
#
# dataset for grid search
X = pd.read_csv('../data/pdeqretsnonan.csv',
                 index_col=0, parse_dates=True).ix[:,:5].resample('M',
                 how='mean').values
Y = np.roll(X,-1,axis=0)

# dataset for decision function visualization
#X_2d = X[:, :2]
#X_2d = X_2d[Y > 0]
#Y_2d = Y[Y > 0]
#Y_2d -= 1

# It is usually a good idea to scale the data for SVM training.
# We are cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the training set and
# just applying it on the test set.

scaler = StandardScaler()

X = scaler.fit_transform(X)

##############################################################################
# Train classifier
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

RS_range = [10,20,100,500,1000]
alpha_range = [0.01, 0.1,0.3,0.6]
param_grid = dict(leakRate=alpha_range, resSize=RS_range)
cv = StratifiedKFold(y=Y, n_folds=3)
grid = GridSearchCV(es.ESNAdapt(), param_grid=param_grid, cv=cv)
grid.fit(X, Y)

print("The best classifier is: ", grid.best_estimator_)

## Now we need to fit a classifier for all parameters in the 2d version
## (we use a smaller set of parameters here because it takes a while to train)
#C_2d_range = [1, 1e2, 1e4]
#gamma_2d_range = [1e-1, 1, 1e1]
#classifiers = []
#for C in C_2d_range:
#    for gamma in gamma_2d_range:
#        clf = SVC(C=C, gamma=gamma)
#        clf.fit(X_2d, Y_2d)
#        classifiers.append((C, gamma, clf))
#
###############################################################################
## visualization
##
## draw visualization of parameter effects
#pl.figure(figsize=(8, 6))
#xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
#for (k, (C, gamma, clf)) in enumerate(classifiers):
#    # evaluate decision function in a grid
#    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#    Z = Z.reshape(xx.shape)
#
#    # visualize decision function for these parameters
#    pl.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
#    pl.title("gamma 10^%d, C 10^%d" % (np.log10(gamma), np.log10(C)),
#             size='medium')
#
#    # visualize parameter's effect on decision function
#    pl.pcolormesh(xx, yy, -Z, cmap=pl.cm.jet)
#    pl.scatter(X_2d[:, 0], X_2d[:, 1], c=Y_2d, cmap=pl.cm.jet)
#    pl.xticks(())
#    pl.yticks(())
#    pl.axis('tight')

# plot the scores of the grid
# grid_scores_ contains parameter settings and scores
score_dict = grid.grid_scores_

# We extract just the scores
scores = [x[1] for x in score_dict]
scores = np.array(scores).reshape(len(RS_range), len(alpha_range))

# draw heatmap of accuracy as a function of gamma and C
pl.figure(figsize=(8, 6))
pl.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
pl.imshow(scores, interpolation='nearest', cmap=pl.cm.spectral)
pl.xlabel('gamma')
pl.ylabel('C')
pl.colorbar()
pl.xticks(np.arange(len(alpha_range)), alpha_range, rotation=45)
pl.yticks(np.arange(len(RS_range)), RS_range)

pl.show()