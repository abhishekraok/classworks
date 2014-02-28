# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:39:26 2014

@author: Abhishek
"""
CharCount = []
InputFile = open('rosalind_dna.txt','r')
InputString = InputFile.read()
ListChars =  ['A', 'C', 'G','T'] #set(InputString)
for i in ListChars:
    CharCount.append(InputString.count(i))

for i in CharCount:
    print(i),