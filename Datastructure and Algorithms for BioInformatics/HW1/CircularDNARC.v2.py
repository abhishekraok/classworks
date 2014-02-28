# -*- coding: utf-8 -*-
"""
@author: Abhishek
"""
import itertools

#circularly rotates string by n
def rotate(strg,n):
    return strg[n:] + strg[:n]

#Gives reverse complement of DNA 
def reverseComplement(InputString):
    OutputString = ''
    for i in InputString:
          if i == 'A':
              OutputString+='T'
          elif i=='T':
              OutputString+='A'
          elif i=='C':
              OutputString+='G'
          elif i=='G':
              OutputString+='C'
    return OutputString[::-1]

#Hamming distance of string a,b
def HamminD(a,b):
    return sum(c1 != c2 for c1, c2 in izip(a,b))

#Finds the largest continuous matching string from start and end for given two string
#Returns length of initial match and end match  .e.g. given ayc and axc ,returns len(a), -len(c)
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

#Finds the rotation for which string a has highest continous match in string b
#Requires len(a) > len(b)
def maxsimilarrotation(a,b):
    N = len(a)
    Endmatchscore, matchscore, rotation, Endrotation = 0,0,0,0
    for j in range(0,N):
        CurrentOpS = rotate(a,j)
        match,EndMatch = Matchstartend(CurrentOpS,b)
        (matchscore, rotation) = max((matchscore, rotation), (match,j))
        (Endmatchscore, Endrotation) = min((Endmatchscore, Endrotation), (EndMatch,j))
    return (matchscore,rotation) if matchscore >= -Endmatchscore else (Endmatchscore,Endrotation)
    

#Main Program
InputString = [line.strip() for line in open('rosalind_gasm.txt')]
OutputString = InputString.pop(2)
kmersize = len(InputString[0])

while True:
    PreviousSize = size(InputString)
    for j in InputString:
        jRC = reverseComplement(j)
        if OutputString.count(j) > 0 or OutputString.count(jRC) > 0:
            InputString.pop(InputString.index(j))
            continue
        if j[:-k]==OutputString[-kmersize+k:]:
            OutputString+= j[-k:]
            InputString.pop(InputString.index(j))
            k=1
            break
        elif jRC[:-k]==OutputString[-kmersize+k:]:
            OutputString+= jRC[-k:]
            InputString.pop(InputString.index(j))
            k=1
            break
    if PreviousSize == size(InputString):
        k += 1
    if k > kmersize/2:
        k=1
        break
    

OutputString=OutputString[:-kmersize+1] #Trimming the last kmer size extra that was added because of circular nature
print OutputString
open('Output.txt','w').write(OutputString)