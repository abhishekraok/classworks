# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 2014
@author: Abhishek
"""
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

OutputCount =[]
temp = ''
reads = []
InputString = [line.strip() for line in open('rosalind_revp.txt')]
for line in InputString:
    if line[0]=='>':
        if temp != '':
            reads.append(temp)
        temp = ''
    else:
        temp += line.strip()
        
if temp != '':
    reads.append(temp)
InputString = reads[0]
N = len(InputString)
for i in range(N):
    for j in range(4,min(N-i+1,13)):
        if InputString[i:i+j] == reverseComplement(InputString[i:i+j]):
            OutputCount.append(str(i+1)+ ' ' +str(j))
for i in OutputCount:
    print i
