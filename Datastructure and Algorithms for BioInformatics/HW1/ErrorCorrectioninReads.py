# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 2014
@author: Abhishek
"""
from itertools import izip

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

def HamminD(a,b):
    return sum(c1 != c2 for c1, c2 in izip(a,b))
    
InputString = [line.strip() for line in open('rosalind_corr.txt')]
OutputString = ''
temp= ''
reads = []

for line in InputString:
    if line[0]=='>':
        if temp != '':
            reads.append(temp)
        temp = ''
    else:
        temp += line.strip()
        
if temp != '':
    reads.append(temp)

for i in reads:
    if (reads.count(i) + reads.count(reverseComplement(i))) < 2:
        for j in reads:
            k = reverseComplement(j)
            if HamminD(i,j) == 1:
                if (reads.count(j) + reads.count(k)) > 1:
                    if OutputString.find(i+ '->' +j + '\n') < 0:
                        OutputString += i+ '->' +j + '\n'
            elif HamminD(i,k) == 1:
                if (reads.count(j) + reads.count(k)) > 1:
                    if OutputString.find(i+ '->' +k + '\n') < 0:
                        OutputString += i+ '->' +k + '\n'
print OutputString