# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 2014
@author: Abhishek
"""
InputString = [line.strip() for line in open('rosalind_gc.txt')]
OutputString = ''
GCContent, GCName, temp= [],[],''

for line in InputString:
    if line[0]=='>':
        if temp != '':
            GCContent.append(float(temp.count('G')+temp.count('C'))*100/len(temp))
        temp = ''
        GCName.append(line.replace('>',''))
    else:
        temp += line.strip()
GCContent.append(float(temp.count('G')+temp.count('C'))*100/len(temp))    

    
print GCName[max( (v, i) for i, v in enumerate(GCContent) )[1]]
print max(GCContent)