# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 13:43:54 2014

@author: Abhishek

Title: Global Alignment with scoring.

Uses Needleman Wunsch algorithm 
Details: http://rosalind.info/problems/glob/?class=104
"""

import ReadFiles as io
import numpy as np

LGC = -5 # Linear Gap cost
Charset = 'A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y _'.split() #Character seet
Costmatrix = [i.split() for i in (open('BLOSUM62.dat','r').read()).splitlines()]

def costchar(a,b):
    return int(Costmatrix[Charset.index(a)][Charset.index(b)])

def costall(a,b):
    """ Alignment cost of string a and b. Using Blosum62 and linear gapcost of LGC"""
    lenA, lenB = len(a), len(b)
    minl, maxl = min(lenA, lenB), max(lenA, lenB)    
    cost = 0
    for i in range(0,minl):  # Compute cost for each character using BLOSUM matrix
        cost += costchar(a[i],b[i])
    return cost + (maxl -minl)*LGC
    
def AlignscoreNW(a,b):
    """ Use Needlman Wunsch algorithm to align string a and b and give its score"""
    lenA, lenB = len(a), len(b)
    minl, maxl = min(lenA, lenB), max(lenA, lenB)
    #Creating Dynamic programming table
    #Init    
    OPT = np.zeros((lenA+1,lenB+1))
    OPT[0] = [i*LGC for i in range(lenB)]
    for i in range(lenA):
        OPT[i][0] = i*LGC
    #Dynamic programing part
    for i in range(1,lenA+1):
        for j in range(1,lenB+1):
            cost1 = costchar(a[i-1],b[j-1]) + OPT[i-1][j-1]
            cost2 = LGC + OPT[i-1][j]     
            cost3 = LGC + OPT[i][j-1]
            OPT[i][j]= max(cost1, cost2, cost3)
    return OPT[lenA][lenB]
    
################## MAIN ############################################
s,t =[i[1] for i in io.getData('FASTA')]
print AlignscoreNW(s,t)