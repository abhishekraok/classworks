# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 21:25:58 2014
@author: Abhishek
"""

import ReadFiles as io

InputString = io.getData('FASTA')
OutputList = []
OutputString = ''
k=3

for i in InputString:
    for j in InputString:
        if i[0] != j[0]:
            if i[1][:k] == j[1][-k:]:
                OutputList.append(i[0]+ ' ' + j[0])

for i in OutputList:
    print i
    OutputString += i + '\n'

open('Output.log','w').write(OutputString)