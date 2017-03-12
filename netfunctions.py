# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 00:26:45 2017

@author: adivt
"""

def parseData(data):
    x=[]
    
    for i in data:
        j = str(i)
        k = j.split("'")
        a.append(k[1]+","+k[3]+","+k[5])
        x.append([a])
    x.pop(0)
    return x
