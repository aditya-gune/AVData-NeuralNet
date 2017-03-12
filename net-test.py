# -*- coding: utf-8 -*-
"""

"""
import re
import numpy
#import netfunctions
#from keras.models import Sequential
#from keras.layers import Dense

#seed
seed = 7
numpy.random.seed(seed)

dataset = numpy.genfromtxt('TestQ.csv',delimiter=',', dtype=None)
#rawdata = parseData(dataset)

x=[]
rawdata=[]
a=[]  
for i in dataset:
    j = str(i)
    k = j.split("'")
    a.append(k[1])
    a.append(k[3])
    a.append(k[5])
    a=[]
    rawdata.append(a)
rawdata.pop(0)

for i in rawdata:
    a.append(i[0])
    a.append(i[1])
    a=[]
    x.append(a)