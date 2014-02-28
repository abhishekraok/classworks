# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 2014
@author: Abhishek
"""
OutputString = ''
InputString = (open('rosalind_revc.txt','r')).read()
for i in InputString:
      if i == 'A':
          OutputString+='T'
      elif i=='T':
          OutputString+='A'
      elif i=='C':
          OutputString+='G'
      elif i=='G':
          OutputString+='C'

print OutputString[::-1]
