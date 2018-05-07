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
import datetime as dt
import time
import os
import threading


CONNECTED = False
BAUD = 9600
PARITY   = ps.PARITY_NONE
STOPBITS = ps.STOPBITS_ONE
BYTESIZE = ps.SEVENBITS
TIMEOUT = 1


thermistor = [0 for x in range(2)]
temperature = [[0 for x in range(2)] for y in range(64)] 

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
    global thermistor
    global temperature
    bytelist = list(bytearray(bytes(data)))
    bytelist.remove(42)
    for i in range(len(bytelist())):
        if (i == 0)|(i == 1)|(i == 2):
            print("header")
        elif i == 3:
            thermistor[0] = bytelist[i]
        elif i == 4:
            thermistor[1] = bytelist[i]
        elif i ==132:
            temperature[1][63] = bytelist[i]
            for j in range(64):
                #Zeichenumrechnung auf Floatzahl
                for k in range(2):
                    s_temp = shAMG_PUB_TMP_ConvTemperature(temperature[i-1][j-1])
                    f_temp = fAMG_PUB_CMN_ConvStoF(s_temp)
                    print("Pixel %2d:  %3.3f" % (j,f_temp))
            s_therm = shAMG_PUB_TMP_ConvThermistor(thermistor);
            f_therm = fAMG_PUB_CMN_ConvStoF(s_therm);
            print("%3.4f,", f_therm);
            print(str(dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")));
            print("\n");
        else:
            temperature[int((i-3) / 2)+((i-3) % 2)][int((i-3)% 2)] = bytelist[i]
           
        # zuordnung der Werte
    
      
                                   
            

def read_from_port(ser, conect):

     while conect:
         reading = ser.readline()
         handle_data(reading)

            

                
if isOSWin10:
    print("Version Lenovo Windows 10")
    PORT = 'COM5'
    
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
 
#thread1 = threading.Thread(target=read_from_port, args=(ser,CONNECTED))
#thread1.daemon = True
#thread1.start()

read_from_port(ser,CONNECTED)
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
        
        




        
        
        