# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 13:43:54 2014

@author: Abhishek

Title: Global Alignment in linear space.

Using Hirschberg algorithm to calculate the global alignment of two strings in O(n) space
Details: http://rosalind.info/problems/5l/
"""

import ReadFiles as io
import numpy as np
import time
import pdb
t0 = time.time()
LGC = -5 # Linear Gap cost
Charset = 'A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y _'.split() #Alphabet seet
Costmatrix = [i.split() for i in (open('BLOSUM62.dat','r').read()).splitlines()] 
costmatrixint = [[np.int16(i) for i in j] for j in Costmatrix]

def costall(a,b):
    """ Alignment cost of string a and b. Using Blosum62 and linear gapcost of LGC"""
    lenA, lenB = len(a), len(b)
    minl, maxl = min(lenA, lenB), max(lenA, lenB)    
    cost = 0
    indexlo = Charset.index
    for i in range(0,minl):  # Compute cost for each character using BLOSUM matrix
        cost += costmatrixint[indexlo(a[i])][indexlo(b[i])]
    return cost + (maxl -minl)*LGC
    
def NW(a,b):
    """ Does Needleman Wunsch string alignment
    NOTE: assumes length of a or b to be < 3"""
    lenA, lenB = len(a), len(b)
    #Simple cases
    if lenA == 1 and lenB == 1:
            return a,b
    elif len(a) == 1:
        indexlo = Charset.index
        pdb.set_trace()
        lastarray = [costmatrixint[indexlo(a)][indexlo(i)] for i in b]
        maxi = lastarray.index(max(lastarray))
        Z = '_'*len(b)
        Z = Z[:maxi]+ a+ Z[maxi+1:]
        return Z,b
    elif len(b) == 1:
        indexlo = Charset.index
        lastarray = [costmatrixint[indexlo(i)][indexlo(b)] for i in a]
        maxi = lastarray.index(max(lastarray))
        Z = '_'*len(a)
        Z = Z[:maxi]+ b+ Z[maxi+1:]
        return a,Z
    else:
        return 'ERRRRRRRRRRRRRRROOOOOOOOOOOOORRR'

def Hirschberg(a,b):
    """Use Hirschberg algorithm to align string a and b"""
    Z = ''
    W = ''
    lenA, lenB = len(a), len(b)
    # Simple cases
    if lenA ==0 or lenB ==0:
        if lenA==0:
            W = b
            Z = '_'*len(b)
        elif lenB==0:
            Z = a
            W = '_'*lenA
    elif lenA==1 or lenB==1:
        Z,W = NW(a,b)
    # Recursion starts
    else:
        mid = lenA/2
        scoreL = AlignscoreNW(a[:mid],b)
        scoreR = AlignscoreNW(a[mid:][::-1],b[::-1])
        sumscore = scoreL + scoreR[::-1]
        ymid = np.where(sumscore == np.amax(sumscore))[0][0]
        Z1,W1 = Hirschberg(a[:mid],b[:ymid])
        Z2,W2 = Hirschberg(a[mid:],b[ymid:]) 
        Z = Z1+Z2
        W = W1+W2
    return Z,W
      
def AlignscoreNW(a,b):
    """ Use Needlman Wunsch algorithm to align string a and b and give its score"""
    lenA, lenB = len(a), len(b)
    #Creating Dynamic programming table
    #Init    
    OPT = np.zeros((lenA+1,lenB+1), dtype=int16)
    OPT[0] = [i*LGC for i in range(lenB+1)]
    for i in range(lenA+1):
        OPT[i][0] = i*LGC
    #Dynamic programing part
    indexlo = Charset.index
    maxlo = max
    for i in xrange(1,lenA+1):
        for j in xrange(1,lenB+1):
            cost1 = costmatrixint[indexlo(a[i-1])][indexlo(b[i-1])] + OPT[i-1][j-1]
            cost2 = LGC + OPT[i-1][j]     
            cost3 = LGC + OPT[i][j-1]
            OPT[i][j]= maxlo(cost1, cost2, cost3)
    return OPT[lenA]
    
################## MAIN ############################################
s,t = io.getData()
a,b = Hirschberg(s,t)
cost = costall(a,b)
print a,b
print cost
open('output.log','w').write(str(cost)+ '\n'+ a+ '\n'+ b)
print (time.time()-t0)/60