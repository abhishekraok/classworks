# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 2014
@author: Abhishek
"""
from itertools import izip
InputString = [line.strip() for line in open('rosalind_hamm.txt')]
print sum(c1 != c2 for c1, c2 in izip(InputString[0],InputString[1]))