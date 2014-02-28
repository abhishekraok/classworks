# -*- coding: utf-8 -*-
"""
Title: Encoding Suffix Trees

Created on Tue Feb 18 20:33:36 2014
@author: Abhishek
Description: From Rosalind.info
Given a string s having length n, recall that its suffix tree T(s) is defined by the following properties:
T(s) is a rooted tree having exactly n leaves.
Every edge of T(s) is labeled with a substring of s∗, where s∗ is the string formed by adding a placeholder symbol $ to the end of s.
Every internal node of T(s) other than the root has at least two children; i.e., it has degree at least 3.
The substring labels for the edges leading down from a node to its children must begin with different symbols.
By concatenating the substrings along edges, each path from the root to a leaf corresponds to a unique suffix of s∗.

Given: A DNA string s of length at most 1kbp.
Return: The substrings of s∗ encoding the edges of the suffix tree for s. You may list these substrings in any order.
"""

import ReadFiles as io

s = io.getData()[0]
Tree = io.buildSuffixTree(s)
OutputList = flattenTree(Tree)
OutputString = ''
for i in OutputList:
    OutputString += i + '\n'
open('Output.log','w').write(OutputString)

