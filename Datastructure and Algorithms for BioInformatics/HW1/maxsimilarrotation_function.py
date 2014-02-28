# -*- coding: utf-8 -*-
"""
Created on Mon Feb 03 21:40:25 2014

@author: Abhishek
"""

def Matchstartend(a,b):
    Match = 0
    for k in range(min(len(a),len(b))):
        if a[Match] != b[Match]:
            break    
        else:
            Match+= 1
    EndMatch = -1
    for k in range(min(len(a),len(b)))[::-1]:
        if a[k] != b[k]:
            break    
        else:
            EndMatch-= 1
    EndMatch+= 1   
    return Match,EndMatch
a='ab'
b='ba'
N = len(a)
Endmatchscore, matchscore, rotation, Endrotation = 0,0,0,0
for j in range(0,N):
    CurrentOpS = rotate(a,j)
    match,EndMatch = Matchstartend(CurrentOpS,b)
    (matchscore, rotation) = max((matchscore, rotation), (match,j))
    (Endmatchscore, Endrotation) = min((Endmatchscore, Endrotation), (EndMatch,j))

    