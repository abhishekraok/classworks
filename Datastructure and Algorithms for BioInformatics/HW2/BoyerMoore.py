# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:50:48 2014

@author: Abhishek
"""
import ReadFiles as io


"""
Main Program starts here
"""
OutputString  = ''
a= io.getData()
T = a[0].strip()
P = a[1].strip()
OutputList = io.BMMatch(T, P)
for i in OutputList:
    print (i),
    OutputString += str(i) + ' '
    
open('Output.log','w').write(OutputString)