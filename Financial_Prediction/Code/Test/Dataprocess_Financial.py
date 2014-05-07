# -*- coding: utf-8 -*-
"""
Created on Tue May 06 17:13:57 2014

@author: Abhishek
"""
import pandas as pd


# Daily with Nan
df= pd.read_csv('data/pdeqrets.csv',index_col=0, parse_dates = True)
df[df.columns[0:3]].plot()
plt.title('Daily return of Company 0,1,2')
plt.xlabel('Year')
plt.ylabel('Percentage change')

#Weekly 
dfn = pd.read_csv('../data/pdeqretsnonan.csv',index_col=0, parse_dates=True)
dfnw= dfn.resample('W',how='mean')
df[df.columns[0]].plot()
plt.ylabel('Percentage change')
plt.xlabel('Year')
plt.title('Weekly returns')

#monthly 
dfnm= dfn.resample('M',how='mean')
dfnm[dfnm.columns[0]].ix[:100].plot(style='d--',label=dfnm.columns[0])
plt.ylabel('Percentage change')
plt.xlabel('Year')
plt.title('Monthly returns')
plt.legend()
