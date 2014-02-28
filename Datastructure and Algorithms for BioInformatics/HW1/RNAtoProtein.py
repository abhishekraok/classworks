# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 2014
@author: Abhishek
"""
OutputString = ''
RNATable = open('RNAcodontable.txt').read()
InputString = open('rosalind_prot.txt','r').read()
N = len(InputString)

for i in range(0,N/3):
    c = RNATable[ RNATable.find(InputString[i*3:i*3+3]) +4]
    if c == '#':
        break
    else:
        OutputString += c
        
print OutputString
