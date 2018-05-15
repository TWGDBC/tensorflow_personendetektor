# -*- coding: utf-8 -*-
"""
Created on Tue May  1 09:47:26 2018

@author: User
"""
import tensorflow as tf
import serial as ps
import datetime as dt
import threading
import statistics as st
import time

## Serial Variablen
CONNECTED = False
BAUD = 19200
PARITY   = ps.PARITY_NONE
STOPBITS = ps.STOPBITS_ONE
BYTESIZE = ps.SEVENBITS
TIMEOUT = 0

# Helper Variablen
thermistor = [0 for x in range(2)]
temperature = [[0 for x in range(2)] for y in range(64)] 
Images = [0.000 for x in range(64)]

# Condition for Synchronisation
stop_ev = threading.Event()

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
    
## Handles serial data and converts data as Image for CNN
def handle_data(data):
    try:   
        global thermistor
        global temperature
        bytelist = list(bytearray(bytes(data)))
        #bytelist.remove(42)
        lock.acquire();
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
                            Images[j] = f_temp 
            elif (i == 133) | (i == 134):
                pass
            else:
                temperature[int((i-5) / 2)+((i-1) % 2)][int((i-5)% 2)] = bytelist[i]
        print(Images[1])
        # here lock release for same ressource
        lock.release()
        # clear event for thread2
        stop_ev.clear()
        #print(str(dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    except KeyboardInterrupt:
        CONNECTED = False
        ser.flush()
        ser.close()         
           
        # zuordnung der Werte

## THREAD1 functon, read serial line    
def read_from_port(ser, connect):

     while connect:
         handle_data(ser.readline())

## THREAD2 functon,     
def calculatePerson():
    try:
        img_size = 8
        # Images are stored in one-dimensional arrays of the length 64.
        img_size_flat = img_size * img_size
        # Number of channels for the images: 1 channel for gray-scale.
        cnt_channels = 1
        # Number of classes, one class for the amount of person [0,1,2,3,4]
        cnt_classes = 5
        # Convolutional Layer 1. 0.9934, 0.9964
        filter_size1 = 3    #5      # Convolution filters are 5 x 5 pixels.
        num_filters1 = 16    #32   # There are 16 of these filters.
        # Convolutional Layer 2.
        filter_size2 = 3   #4       # Convolution filters are 3 x 3 pixels.
        num_filters2 = 24    #16    # There are 36 of these filters.
        # Convolutonal Layer 3.
        filter_size3 = 3   #2       # Convolution filters are 3 x 3 pixels.
        num_filters3 = 32    #8   # There are 36 of these filters.
        #fully-connected layer size --> fc_size klein halten, Exponential features
        fc_size = 96 # 128
        
        Pixels = tf.placeholder(tf.float32, shape=[img_size_flat], name='Pixels')
        # batch must be 4 Dimensions, so reshape
        Pixel_image = tf.reshape(Pixels, [-1 ,img_size, img_size, cnt_channels], name='Pixel_Image')
        # 
        cnt_true = tf.placeholder(tf.float32, shape=[None, cnt_classes], name='cnt_true')
        # compare
        cnt_true_cls = tf.argmax(cnt_true, axis=1)

        # create a layer implementation
        pixel_input = Pixel_image
        # conv layer 1
        conv1 = tf.layers.conv2d(inputs=pixel_input, name='layer_conv1', padding='same',
                                 filters=num_filters1, kernel_size=filter_size1, activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=1)
        ###############################################################################
        # conv layer 2
        conv2 = tf.layers.conv2d(inputs=pool1, name='layer_conv2', padding='same',
                                 filters=num_filters2, kernel_size=filter_size2, strides = 2, activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=1)
        ###############################################################################
        ## Eventuell nötig
        conv3 = tf.layers.conv2d(inputs=pool2, name='layer_conv3', padding='same',
                                 filters=num_filters3, kernel_size=filter_size3, activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=1)
        ###############################################################################
        # flatten layer
        flatten = tf.layers.flatten(pool3)
        # Fully connected layer
        fc1 = tf.layers.dense(inputs=flatten, name='layer_fc1',
                              units=fc_size, activation=tf.nn.relu)
        fc2 = tf.layers.dense(inputs=fc1, name='layer_fc_out',
                              units=cnt_classes, activation=None)
        
        logits = fc2
        cnt_pred = tf.nn.softmax(logits=logits)
        # compare
        cnt_pred_cls = tf.argmax(cnt_pred, axis=1)
        
        saver = tf.train.Saver(max_to_keep= 200)
        restore_path = 'C:/Users/User/switchdrive/HSLU_6_Semester\BAT/projects/Tensorflow/tmp/model_2018-05-14_09-49-38.ckpt'
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver.restore(sess=session, save_path=restore_path)       
        prediction = []
        i = 0
        while True:
            while stop_ev.is_set():
                stop_ev.wait(0.01)
            lock.acquire()
            feed_dict = {Pixels: Images} 
            lock.release()
            # Calculate the predicted class using TensorFlow.
            cls_pred = session.run(cnt_pred_cls, feed_dict=feed_dict)
            print(cls_pred)
            #print(str(dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
#            i += 1
            ## stop event for synchronisation
            stop_ev.set()
#            prediction.append(cls_pred)
#            if (i == 10):
#                print(st.median(prediction))
#                print(str(dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
#                 list.clear
#                 i = 0
            
    except KeyboardInterrupt:
        CONNECTED = False
        if thread1.is_alive:
            thread1._stop
        if thread2.is_alive:
            thread2._stop
        if (ser.closed == True):
            ser.flush()
            ser.close()
                    

    
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


## INIT THREADS
thread1 = threading.Thread(target=read_from_port, args=(ser, CONNECTED))
thread1.daemon = True

thread2 = threading.Thread(target=calculatePerson, args=())
thread2.daemon = True
lock = threading.Lock()

thread1.start()
thread2.start()

try:
    while(CONNECTED):
        pass
        
except KeyboardInterrupt:
    CONNECTED = False
    if (ser.closed== False):
        ser.flush()
        ser.close()
finally:
    if CONNECTED == True:
        CONNECTED = False
    ser.close()
    if thread1.is_alive:
        thread1._stop
    if thread2.is_alive:
        thread2._stop
        
        




        
        
        