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

edges = []
InputString = open('rosalind_dbru.txt','r')
for line in InputString:
    k = len(line.strip())
    edges.append((line[0:k-1],line[1:k]))
    revline = reverseComplement(line.strip())
    edges.append((revline[0:k-1],revline[1:k]))
    break

for line in InputString:
    edges.append((line[0:k-1],line[1:k]))
    revline = reverseComplement(line.strip())
    edges.append((revline[0:k-1],revline[1:k]))

EdgeList = sorted(list((set(edges))))

for i in EdgeList:
    print '(' +i[0] +', ' +i[1] +')'