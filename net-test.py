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

x=[]
y=[]
rawdata=[]
a=[]  

#parse data
for i in dataset:
    j = str(i)
    k = j.split("'")
    a.append(k[1])  #event id
    a.append(k[3])  #seq of events
    a.append(k[5])  #class label
    a=[]
    rawdata.append(a)

rawdata.pop(0)  #remove first element (column headers)
rawdata.pop()   #remove last element (empty list)

#dedupe list, concatenate eventseqs for common events
for i in rawdata:
    for j in rawdata:
        if i[0] == j[0]:
            i[1] += " " + j[1]

            del rawdata[rawdata.index(j)]
            
numpy.random.shuffle(rawdata)   #shuffle rawdata            

#create X and Y matrices
for i in rawdata:
    a=[]
    a.append(i[0])
    a.append(i[1])
    x.append(a)
    y.append([i[2]])