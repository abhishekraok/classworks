# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:11:30 2014

@author: Abhishek

Finds all the maximal repeats in the string. See http://rosalind.info/problems/mrep/

Overview
1. Read the input string
2. Create Suffix Tree
3. Traverse the tree and list all nodes that satisfy the condition
4. Remove substrings from the list 
"""

import ReadFiles as io

MIN_DEPTH = 20  # Mininum length of accepted strings.

def number_of_children(T):
    """ Gives the number of nodes in the tree T. """
    return traverse_tree(T, sum, 1)

def traverse_tree(Tree, func, base_value):
    """ Traverses a tree inorder and does func() to each edge while traversing
    Returns base value in the case of tree being []
    """
    if Tree == []:
        return base_value
    else:
        return func([traverse_tree(i[1], func, base_value) for i in Tree])
    
def Depth_Checker(Tree, L=[], currentpath=''):
    """ Traverses through Tree, checkes each node for condition if depth of 
    current node > 20 and number of children > 1. If so appends it to list L.    
    """
    if Tree == []:
        return L
    # Remember currentsize
    #CurrentListSize = len(L)  
    # Check if current node satisfies (depth  > 20 and children > 2)
    #if yes (path, no of children) to list
    # continue doing same for each children        
    children = number_of_children(Tree)
    if len(currentpath) >= MIN_DEPTH and children > 1:
        L.append([currentpath, children])     
    # Do the same for children
    for i in Tree:
        #Recursion step, do the same for each child    
        L = Depth_Checker(i[1], L, currentpath+ i[0])
    # Write to list only if none of the children wrote, else don't write we don't want to write 
    # for every path > 20
    #if CurrentListSize == len(L) or True:    
    return L

def remove_substrings(L):
    """ Removes i from L if i is substring of j !i in L 
    For each repeat value. 
    Input data is of the format [ [string1, count1], [string2, count2], ...]
    """

    Li = []
    RepCount = list(set([i[1] for i in L]))
    # separte different counts substring into seperate list.     
    for M in RepCount:
        Li.append([i for i in L if i[1]==M])
    # List to return
    retlist = []
    for m in Li:
        # This complicated list comprehension returns those elements of L that 
        # satisfy the condition that they are not substring of any other.
        retlist += [m[k] for k in [i for i,j in enumerate([sum([j[0] in i[0] for i in m]) for j in m]) if j ==1]]
    return retlist         
    
def findmaximal(L):
    """ Given a list of [path, number of children], find one with maximum no. of children
    and return all those lists
    """
    if L == []:
        return []
    M = max(L, key = lambda x: x[1]) 
    maxindices = [i for i, j in enumerate(L) if j[1] == M[1]]
    return [L[i] for i in maxindices]
    
####################### Main ##############################
s = io.getData()[0]+'$'
Tree = io.buildSuffixTree(s)
Candidates = Depth_Checker(Tree)
OutputList = remove_substrings(Candidates)
OutputString = ''
for i in OutputList:
    OutputString += i[0] + '\n'
open('Output.log','w').write(OutputString)