# -*- coding: utf-8 -*-
"""
Created on Tue May  1 09:47:26 2018

@author: User
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import serial as ps
from array import array
import time
import os
import threading


CONNECTED = False
BAUD = 9600
PARITY   = ps.PARITY_NONE
STOPBITS = ps.STOPBITS_ONE
BYTESIZE = ps.SEVENBITS
TIMEOUT = 1

# check if you use RPI or Anaconda Win10
isOSWin10 = True


# Helperfunction to convert Thermistor Temperature
def shAMG_PUB_TMP_ConvThermistor(aucRegVal):
    shVal = ((aucRegVal[1] & 0x07) << 8) | aucRegVal[0]
    if (0 != (0x08 & aucRegVal[1])):
        shVal *= -1
    shVal *= 16;
    return (shVal);

# Helperfunction to convert PIXEL Temperature
def shAMG_PUB_TMP_ConvTemperature(aucRegVal):
    shVal = ((aucRegVal[1] & 0x07) << 8) | aucRegVal[0]
    if (0 != (0x08 & aucRegVal[1])):
        shVal -= 2048
    shVal *= 64
    return (shVal)

# Helperfunction to convert Value
def fAMG_PUB_CMN_ConvStoF(shVal):
    return ((shVal)/(float(256)))


## THREADS Helperfunction
def handle_data(data):
      
        bytelist = list(bytearray(bytes(data))
        for i in range(len(bytelist)):
            if i == 0: 
        
            if i 
            


def read_from_port(ser, conect):

     while conect:
         reading = ser.readline()
         handle_data(str(reading))

            

                
if isOSWin10:
    print("Version Lenovo Windows 10")
    PORT = 'COM10'
    
    ser = ps.Serial(
            port=PORT,\
            baudrate=BAUD,\
            parity= PARITY,\
            stopbits= STOPBITS,\
            bytesize= BYTESIZE,\
            timeout=TIMEOUT)
    print("connected to: " + ser.portstr)
    CONNECTED = True
        
else:
    print("Raspbery Pi 3 Raspian Jessie")
    PORT ='/dev/ttyUSB0'
    
#    reading_event = thread.Event()
#    reading_thread = thread.Thread(target=reading, daemon=True)
    
#    def reading():
#        while reading_event.is_set():
#            raw_reading = ser.readline()
#            print(raw_reading)
    ser = ps.Serial(
            port=PORT,\
            baudrate=BAUD,\
            parity= PARITY,\
            stopbits= STOPBITS,\
            bytesize= BYTESIZE,\
            timeout=TIMEOUT)
    print("connected to: " + ser.portstr)
 
thread1 = threading.Thread(target=read_from_port, args=(ser,CONNECTED))
thread1.daemon = True
thread1.start()

try:
    while(1):
        time.sleep(1)
except KeyboardInterrupt:
    CONNECTED = False
    ser.flush()
    ser.close()
finally:
    if CONNECTED == True:
        CONNECTED = False
    ser.flush()
    ser.clos()
        
        




        
        
        