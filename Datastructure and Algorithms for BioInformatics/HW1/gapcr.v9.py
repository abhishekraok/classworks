# -*- coding: utf-8 -*-
"""
@author: Abhishek
"""
def getstringfromlist(InputStringList):
    k = len(InputStringList[0])
    return InputStringList[0]+ ''.join([x[-1] for x in InputStringList[1:-k+1]])
    
def swap(list,i,j):
    (i,j) = (min(i,j),max(i,j))
    return list[:i] + list [j:] + list[i:j]
    
    
#Main Program
InputStringList = [line.strip() for line in open('rosalind_gasm.txt')]
Outputlist = []

for i in range(size(InputStringList)):
    #Check for multiple same first entries
    if sum([1 if InputStringList[i][:-1] == x[:-1] else 0 for x in InputStringList]) >1 :
        for j in range(size(InputStringList)):
            if InputStringList[i][:-1]== InputStringList[j][:-1]:
                newlist = swap(InputStringList,i,j)
                newoutput = getstringfromlist(newlist)
                if Outputlist.count(newoutput) < 1:
                    Outputlist.append(newoutput)

print Outputlist