# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 19:47:01 2014

@author: Abhishek
"""

Text =  (open('rosalind_iprb.txt','r')).read()
a = Text.split(' ')
for i in Text:
    a.append(i)
k = float(a[0].strip())
m = float(a[1].strip())
n = float(a[2].strip())
#m,n,k = 2.0,2.0,2.0
Total = m + n + k
ProbRecessive = (n/Total)*(n-1)/(Total-1) + 0.5*(m/Total)*n/(Total-1) + 0.5*n/Total*m/(Total-1) + 0.25*(m/Total)*(m-1)/(Total-1)
print (1-ProbRecessive)