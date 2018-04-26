# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 23:28:54 2018

@author: User
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils

DATA_FILE = '

# Step 1: read in the data
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Step 2: create Dataset and iterator
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))

iterator = dataset.make_initializable_iterator()
P1, P2, P3, P4, P5,, P6, P7, P8, P9, P10,  P11 = iterator.get_next()









def read_Pixel_data(filename):
    """
    
    data in the form of NumPy array
    n_samples: number of samples
    """
    text = open(filename, 'r').readlines()[1:]
    data = [line[:-1].split('\t') for line in text]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    births = [float(line[3]) for line in data]
    lifes = [float(line[4]) for line in data]
    births = [float(line[5]) for line in data]
    lifes = [float(line[6]) for line in data]
    births = [float(line[7]) for line in data]
    lifes = [float(line[8]) for line in data]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    data = list(zip(births, lifes))
    n_samples = len(data)
    data = np.asarray(data, dtype=np.float32)
    return data, n_samples