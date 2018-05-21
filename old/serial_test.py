# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:42:59 2018

@author: User
"""

import serial as ps
import datetime as dt
import time
import numpy as np


ser = ps.Serial('COM5', baudrate= 9600, bytesize= ps.EIGHTBITS, timeout = 5)

try:
    while 1:
        #print("---")
        a = bytearray(bytes(ser.readline()))
        z= list(a)
        i = 0
        #a = ser.readline()
        #print(a)
        print(str(dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        print(z)
        print(ser.inWaiting())
        #for i in range(len(z)):
         #  print((z[i]))


        
except KeyboardInterrupt:
    ser.close ()       