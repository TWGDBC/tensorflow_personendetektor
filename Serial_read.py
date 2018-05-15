# -*- coding: utf-8 -*-
"""
Created on Tue May  1 09:47:26 2018

@author: User
"""
import tensorflow as tf
import serial as ps

import datetime as dt
import threading


CONNECTED = False
BAUD = 9600
PARITY   = ps.PARITY_NONE
STOPBITS = ps.STOPBITS_ONE
BYTESIZE = ps.SEVENBITS
TIMEOUT = 0


thermistor = [0 for x in range(2)]
temperature = [[0 for x in range(2)] for y in range(64)] 
Pixels = [0.000 for x in range(64)]

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
    try:   
        global thermistor
        global temperature
        bytelist = list(bytearray(bytes(data)))
        #bytelist.remove(42)
        lock.acquire()
        for i in range(len(bytelist)):
            if (i == 0)|(i == 1)|(i == 2):
                pass
                #print("header")
            elif i == 3:
                thermistor[0] = bytelist[i]
            elif i == 4:
                thermistor[1] = bytelist[i]
            elif i ==132:
                temperature[63][1] = bytelist[i]
                for j in range(64):
                    #Zeichenumrechnung auf Floatzahl
                    for k in range(2):
                        #Pixelwerte umwandeln
                        #s_temp = shAMG_PUB_TMP_ConvTemperature(temperature[j])
                        #f_temp = fAMG_PUB_CMN_ConvStoF(s_temp)
                        f_temp = fAMG_PUB_CMN_ConvStoF(shAMG_PUB_TMP_ConvTemperature(temperature[j]))                     
                        #print("Pixel %2d:  %3.3f" % (j,f_temp))
                        
                        #thermistorwerte umwandeln           
                        #s_therm = shAMG_PUB_TMP_ConvThermistor(thermistor);
                        #f_therm = fAMG_PUB_CMN_ConvStoF(s_therm);
                        #f_therm = fAMG_PUB_CMN_ConvStoF(s_therm);
                        
                        #print(Thermistor "%3.4f,", f_therm);
                        
                        if (k == 1):
                        
                            Pixels[j] = f_temp 
            elif (i == 133) | (i == 134):
                pass
                                
            else:
                temperature[int((i-5) / 2)+((i-1) % 2)][int((i-5)% 2)] = bytelist[i]
        #print(Pixels)
        lock.release()
        print(str(dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    except KeyboardInterrupt:
        CONNECTED = False
        ser.flush()
        ser.close()         
           
        # zuordnung der Werte
    
def read_from_port(ser, connect):

     while connect:
         handle_data(ser.readline())

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
 
thread1 = threading.Thread(target=read_from_port, args=(ser, CONNECTED))
thread1.daemon = True
lock = threading.Lock()


thread1.start()
#thread2.start()

try:
    while(CONNECTED):
        pass
        
except KeyboardInterrupt:
    CONNECTED = False
    ser.flush()
    ser.close()
finally:
    if CONNECTED == True:
        CONNECTED = False
    ser.close()
    thread1._stop
        
        




        
        
        