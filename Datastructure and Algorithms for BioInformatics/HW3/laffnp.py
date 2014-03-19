# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 16:27 2014

@author: Abhishek

Title: Local alignment with affine penalty gap

Uses Needleman Wunsch algorithm , using numpy
Details: http://rosalind.info/problems/laff
"""
import ReadFiles as io
import numpy as np
import time

def traceback(A, i, j):
    """ Traceback in the NW algorithm"""
    movediag = (A[i-1][j-1],  i-1, j-1)
    moveside = (A[i][j-1],    i,   j-1)
    movedown = (A[i-1][j],    i-1, j)
    return (max(movediag,moveside,movedown)[1:])

################## MAIN ############################################
t0 = time.time()
AGCO = np.int16(-11) # Affine gap opening cost
AGCE = np.int16(-1)  # Extension cost
Charset = 'A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y _'.split() #Character seet
Costmatrixs = [i.split() for i in (open('BLOSUM62.dat','r').read()).splitlines()]
costmatrixint = [[np.int16(i) for i in j] for j in Costmatrixs]
a,b =[i[1] for i in io.getData('FASTA')]
lenA, lenB = len(a), len(b)
minl, maxl = min(lenA, lenB), max(lenA, lenB)
#Creating Dynamic programming table
#Init     
M = np.zeros((lenA+1,lenB+1),dtype=np.int16)
X = np.zeros((lenA+1,lenB+1),dtype=np.int16)
Y = np.zeros((lenA+1,lenB+1),dtype=np.int16)

for i in range(lenA+1):
    M[i][0] = -3000
    X[i][0] = -3000
    Y[i][0] = AGCO + i*AGCE
for i in range(lenB+1):
    M[0][i] = -3000
    X[0][i] = AGCO + i*AGCE
    Y[0][i] = -3000
#Dynamic programing part
for i in range(1,lenA+1):
    for j in range(1,lenB+1):
        M[i][j] = costmatrixint[Charset.index(a[i-1])][Charset.index(b[j-1])]\
                    + max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1], 0)
        X[i][j] = max(\
                    AGCO+ AGCE+ M[i][j-1],\
                    AGCE +X[i][j-1],\
                    AGCO+ AGCE+ Y[i][j-1],\
                    0\
                    )
        Y[i][j] = max(\
                    AGCO+ AGCE+ M[i-1][j],\
                    AGCE +Y[i-1][j],\
                    AGCO+ AGCE+ X[i-1][j],\
                    0\
                    )
        
print (time.time()-t0)/60       
biggestmat = max([[ np.amax(i),i] for i in [M,X,Y] ])
imax, jmax = np.where(biggestmat[0] == biggestmat[1])
# Tracebak till that matrix element is 0
k,l = traceback(biggestmat[1], np.int16(imax[0]), np.int16(jmax[0]))
while biggestmat[1][k][l] > 0 :
    i,j = k,l
    k,l = traceback(biggestmat[1], k, l)

open('output.log','w').write(str(biggestmat[0])+ '\n'+ a[i-1:imax] + '\n'+ b[j-1:jmax])
print (time.time()-t0)/60