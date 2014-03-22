# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 16:27 2014

@author: Abhishek

Title: Local alignment with affine penalty gap

Uses Needleman Wunsch algorithm , using numpy.
Caching version.
Details: http://rosalind.info/problems/laff
"""
import ReadFiles as io
import numpy as np

def traceback(A, i, j):
    """ Traceback one step in the NW algorithm. 
	TODO: Need to consider all three matrix"""
    movediag = (A[i-1][j-1],  i-1, j-1)
    moveside = (A[i][j-1],    i,   j-1)
    movedown = (A[i-1][j],    i-1, j)
    return (max(movediag,moveside,movedown)[1:])
    
def laff(a,b):
    """Local affine gap matrix creation function. Returns the matrices M,X,Y"""
    AGCO = np.int16(-11) # Affine gap opening cost
    AGCE = np.int16(-1)  # Extension cost
    AGCEO = np.int16(-12)
    Charset = 'A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y _'.split() #Character seet
    Costmatrixs = [i.split() for i in (open('BLOSUM62.dat','r').read()).splitlines()]
    costmatrixint = [[np.int16(i) for i in j] for j in Costmatrixs]
    
    lenA, lenB = len(a), len(b)
    minl, maxl = min(lenA, lenB), max(lenA, lenB)
    #Creating Dynamic programming table
    #Init     
    M = np.zeros((lenA+1,lenB+1),dtype=np.int16)
    X = np.zeros((lenA+1,lenB+1),dtype=np.int16)
    Y = np.zeros((lenA+1,lenB+1),dtype=np.int16)
    prM = np.zeros(lenB+1,dtype=np.int16)
    prX = np.zeros(lenB+1,dtype=np.int16)
    prY = np.zeros(lenB+1,dtype=np.int16)
    tM = np.zeros(lenB+1,dtype=np.int16)
    tX = np.zeros(lenB+1,dtype=np.int16)
    tY = np.zeros(lenB+1,dtype=np.int16)
    maxlo = max
    indexlo = Charset.index    
    for i in range(lenA+1):
        M[i][0] = -3000
        X[i][0] = -3000
        Y[i][0] = AGCO + i*AGCE
    for i in range(lenB+1):
        M[0][i] = -3000
        X[0][i] = AGCO + i*AGCE
        Y[0][i] = -3000
    # All these prM,prX, pcM etc are used in an attempt to cache the data and minimize IO by
	# not using the big M,X, Y matrix frequenly.
    prM = M[0]
    prX = X[0]
    prY = Y[0]
    pcM, pcX, pcY = prM[0], prX[0], prY[0]
    #Dynamic programing part
    for i in xrange(1,lenA+1):
        for j in xrange(1,lenB+1):
            pcMn = costmatrixint[indexlo(a[i-1])][indexlo(b[j-1])]\
                 + maxlo(
                        prM[j-1], 
                        prX[j-1], 
                        prY[j-1], 
                        0
                        )
            pcXn = maxlo(
                        AGCEO+ pcM,
                        AGCE + pcX,
                        AGCEO+ pcY,
                        )
            pcYn = maxlo(
                        AGCEO+ prM[j],
                        AGCE + prY[j],
                        AGCEO+ prX[j],
                        )
            tM[j], tX[j], tY[j] = pcMn, pcXn, pcYn
            pcM, pcX, pcY = pcMn, pcXn, pcYn
        #Update the row
        prM, prX, prY = M[i], X[i], Y[i]
        M[i], X[i], Y[i] = tM, tX, tY
    return M,X,Y
    
################## MAIN ############################################
a,b =[i[1] for i in io.getData('FASTA')]
M,X,Y = laff(a,b)
biggestmat = max([[ np.amax(i),i] for i in [M,X,Y] ])
imax, jmax = np.where(biggestmat[0] == biggestmat[1])
# Tracebak till that matrix element is 0
k,l = traceback(biggestmat[1], np.int16(imax[0]), np.int16(jmax[0]))
i,j = 1,1
while biggestmat[1][k][l] > 0 :
    i,j = k,l
    k,l = traceback(biggestmat[1], k, l)

open('output.log','w').write(str(biggestmat[0])+ '\n'+ a[i-1:imax] + '\n'+ b[j-1:jmax])
