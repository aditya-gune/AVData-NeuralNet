# -*- coding: utf-8 -*-
"""
ADITYA GUNE
"""
import sys
import re
import numpy
from collections import Counter
import string

pelican = int(sys.argv[1])
if pelican > 0:
    from keras.preprocessing import sequence
    from keras.models import Sequential
    from keras.models import load_model
    from keras.layers import LSTM
    from keras.layers.embeddings import Embedding
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    #from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
    from keras.optimizers import SGD
    from keras.utils import np_utils

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

longest_sample_len = len(max(x,key=len))
if pelican > 0:
    """
    NEURAL NETWORK STARTS HERE
    """
    x = sequence.pad_sequences(x, maxlen=longest_sample_len)

    model = Sequential()
    
    #add layers
    model.add(Embedding(6000, 32))
    model.add(LSTM(32))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    # Fit the model
    print(model.summary())
    model.fit(x, y, nb_epoch=40, batch_size=10)
    
    # evaluate the model
    scores = model.evaluate(x, y)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    

