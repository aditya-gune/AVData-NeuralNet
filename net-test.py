# -*- coding: utf-8 -*-
"""

"""
import re
import numpy
from collections import Counter
import string
#import netfunctions
#from keras.models import Sequential
#from keras.layers import Dense

#seed
seed = 7
numpy.random.seed(seed)

dataset = numpy.genfromtxt('TestQ.csv',delimiter=',', dtype=None)

x=[]
xwords=[]
y=[]
rawdata=[]
a=[]  
temp=[]
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
    #a.append(i[0])
    a.append(i[1])
    xwords.append(a)
    y.append([i[2]])

#temp =numpy.array(xwords)
#temp = temp.tolist()

str1=''
for i in xwords:
    i = str(i).translate(None, string.punctuation)
    temp.append(i)
    str1 += ' '+i
    

counts = Counter(str1.split())
numwords = len(counts)
dict={}
counter = 100
for i in counts:
    dict[i] = counter
    counter += 1

for i in temp:
    a = i.split()
    b = []
    for j in a:
        b.append(dict[j])
    x.append(b)