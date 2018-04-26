# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 23:18:29 2018

@author: User
"""


# Rescale data (between 0 and 1)
import pandas
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler

dataframe = pandas.read_csv(, names=names)

cnt_files = 100;



# Reshape and normalize training data
trainX = trainSet[:, 1:].reshape(train.shape[0], 1, 8, 8).astype( 'float32' )
testX = test.reshape(test.shape[0], 1, 8, 8).astype( 'float32' )


    


# One-hot encode output variable
trainY = np_utils.to_categorical(train[:, 0])
num_classes = trainY.shape[1]







names = ['test', 'train', 'validate',]

filenameArray = np.zeros

np.loadtxt(")

Pixels = np.empty(3,2)



array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])


def getDataFromFile:
    
    Pixel = np.array_str[64]
    
    
    
    return Pixel, Thermistor, TimeStamp,


def createNames:
    
    int i;
    Pixel = np.array_str[64]
    for i in range (1, 64):
        Pixel[i] = 'Pixel'+str(i)
    
    return Names



