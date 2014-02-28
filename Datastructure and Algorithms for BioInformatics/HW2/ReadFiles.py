# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 20:28:08 2014

@author: Abhishek
"""
import os
import copy

def addedge_to_sfxtree(L,e):
    """
    Adds the new suffix 'e' to the list of edges in an already existing suffix tree L
    Returns updated suffix tree L'    
    
    Tree datastructure is as folllows: A list of list. Each element is an edge + subtree.
    Tree = [ [edge1, subtree1], [ege2, subtree2], ..]
    """
    Matchfound = 0
    if L == []:
        return [[e,[]]]
    for branch in L:
        bl = len(branch[0])
        ml = LenLongMatch(branch[0],e)
        #Prevent empty edges
        if branch[0] == '':
            continue
        if ml == 0:
            continue
        elif ml == bl:
            #Go one level down
            newbranch = copy.deepcopy(branch)
            L.append( [newbranch[0],  addedge_to_sfxtree(newbranch[1],e[bl:])])
            L.remove(branch)
            Matchfound = 1
            break
        else:
            newbranch = copy.deepcopy(branch)
            L.append(splitbranch(branch[0], branch[1], e[ml:], ml))
            L.remove(newbranch)
            Matchfound = 1
            break
    if Matchfound == 0:
        L.append([e,[]])
    return L

def flattenTree(T, F=[]):
    """
    Prints the nodes of tree inorder by concatenating it to s
    """
    if T == []:
        return T
    for i in T:
        F.append(i[0])
        if i[1] != []:
            F = flattenTree(i[1], F)
    return F

def splitbranch(edge,L,insedge,ml):
    """
    Split the branch. Now the edge is split 0:ml and ml:bl
    The current branches new egdge = oldedge[0:ml] and it's subtreee contains
    """ 
    newbranch = edge[0:ml]
    newchildbranch1 = edge[ml:]
    newchildtree1 = L
    newchildbranch2 = insedge
    newsubtree = [[newchildbranch1, newchildtree1], [newchildbranch2,[]] ]
    return [newbranch, newsubtree]

def buildSuffixTree(s):
    """ Builds a suffix tree from the given string.
    """
    n = len(s)
    Tree = []
    for i in range(0,n):
        Tree = addedge_to_sfxtree(Tree,s[i:])
    return Tree

def getFirstTxtFile():
    """
    Returns the name of the first text file in the current directory.
    """
    for path, subdirs, files in os.walk('.'):
        for filename in files:
            if filename[-4:] == '.txt':	# Checking if the extension is .txt
                return filename
    return ''

def getFASTAstr(InputString):
	#Converts InputString in FASTA format to a list of [ (name,string) ] format
    temp, name = '',''
    OutputList = []
    i = 0
    while i < len(InputString):
        if InputString[i][0]=='>':
            temp = ''
            name = InputString[i].replace('>','')
            i += 1
            while i < len(InputString) and InputString[i][0] !='>':
                temp += InputString[i].strip()
                i += 1
            OutputList.append([name,temp])
    return OutputList
        
        
def getData(fmt = 'string'):
    """
    Main function to be called which calls other function. Reads the first text file and processes the string.
    'fmt' specifies whether it is FASTA or not, leave blank for default normal read.
    """
    fh = open(getFirstTxtFile())
    InputString = [line.strip() for line in fh]
    if fmt == 'string':
        OutputString = InputString
    elif fmt == 'FASTA':
        OutputString = getFASTAstr(InputString)
    return OutputString


def BadCharSkip(T,P,i,j):
    initi = i
    while T[i+j] != P[j]:
        if i > len(T) - len(P):
            return i - initi
        i += 1
    return i - initi
    
def BMMatch(T, P):
    OutputList = []
    #Preprocessing
    L = findLi(P) # For goodshfift rule
    n = len(P)
    i = 0
    try:
        while i < len(T) - n + 1:
            j = n-1
            while j >= 0 and T[i+j] == P[j]:
                j -= 1
            if j == -1:
                OutputList.append(i+1)
                i += 1
            else:
                step = max(goodshift(n,j,L), BadCharSkip(T,P,i,j) ,1)
                i += step
    except IndexError: 
        print i
    return OutputList

def LenLongMatch(a,b):
    """
    Gives the length of the longest continuous prefix match between a and b. 
    i.e. returns k where k is the max satisfying a[:k] = b[:k]
    """
    i=0
    while  i < min(len(a),len(b)):
        if a[i] == b[i]:
            i += 1
        else:
            break
    return i

def findLi(P):
    """
    Calculates the Li needed for Boyer Moor Algorithm's Good shift rule.
    """
    n = len(P)
    zr = calcZi(P[::-1])
    N,L = [0]*n, [0]*n
    for j in range(0,n) :
        N[j] = zr[n- j -1]
    print N
    for j in range(0,n):
        i = n -N[j] -1
        L[i] = j
    return L
    
def calcZi(P):
    """
    Calculates the Zi needed for Boyer Moor Algorithm's Good shift rule.
    """
    z = [0]*len(P)
    for i in range(1,len(P)):
        z[i] = LenLongMatch(P[:i],P[i:])
    return z

def goodshift(n,i,L):
    """ Good shift rule in the Boyer Moor Algorithm. """
    return  n - L[i] -2