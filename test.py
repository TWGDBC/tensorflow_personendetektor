# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 23:28:54 2018

@author: User
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf

train_path = 'input_data/m_a_output_2018-03-28_18-13-28.csv'
test_path =  'input_data/m_a_output_2018-03-28_18-13-28.csv'

CSV_COLUMN_NAMES = ['Pixel 0' ,'Pixel 1' ,'Pixel 2' ,'Pixel 3' ,'Pixel 4' ,'Pixel 5' ,'Pixel 6' ,'Pixel 7' ,'Pixel 8' ,
'Pixel 9' ,'Pixel 10' ,'Pixel 11' ,'Pixel 12' ,'Pixel 13' ,'Pixel 14' ,'Pixel 15' ,'Pixel 16' ,
'Pixel 17' ,'Pixel 18' ,'Pixel 19' ,'Pixel 20' ,'Pixel 21' ,'Pixel 22' ,'Pixel 23' ,'Pixel 24' ,
'Pixel 25' ,'Pixel 26' ,'Pixel 27' ,'Pixel 28' ,'Pixel 29' ,'Pixel 30' ,'Pixel 31' ,'Pixel 32' ,
'Pixel 33' ,'Pixel 34' ,'Pixel 35' ,'Pixel 36' ,'Pixel 37' ,'Pixel 38' ,'Pixel 39' ,'Pixel 40' ,
'Pixel 41' ,'Pixel 42' ,'Pixel 43' ,'Pixel 44' ,'Pixel 45' ,'Pixel 46' ,'Pixel 47' ,'Pixel 48' ,
'Pixel 49' ,'Pixel 50' ,'Pixel 51' ,'Pixel 52' ,'Pixel 53' ,'Pixel 54' ,'Pixel 55' ,'Pixel 56' , 
'Pixel 57' ,'Pixel 58' ,'Pixel 59' ,'Pixel 60' ,'Pixel 61' ,'Pixel 62' ,'Pixel 63' ,'Thermistor' ,'Timestamp']


test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
test2 = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
# delekte Timestamp & Thermistor
test.pop("Timestamp")
test.pop('Thermistor')
test2.pop("Timestamp")
test2.pop('Thermistor')

labels = pd.DataFrame(data = np.ones((64,1)), columns=['labels'])
test3 = concat = pd.concat([test,test2],axis = 0)
test3.drop_duplicates()





#s = set()
#for i in range(0,596):
#    tmp = test.loc[i]['Pixel 0':'Pixel 63']
#    if tmp in s:
       
        
