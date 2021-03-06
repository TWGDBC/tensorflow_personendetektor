# -*- coding: utf-8 -*-
"""
Created on Sat May  5 23:35:58 2018

@author: User
"""
# Import classes and functions
import tensorflow as tf
import numpy as np
from datetime import timedelta
import datetime
import matplotlib.pyplot as plt
import math
import time
import input_data

saver = tf.train.Saver(max_to_keep= 100)
restore_path = 'C:/Users/User/switchdrive/HSLU_6_Semester/BAT/projects/tensorflow_personendetektor/tmp/model_2018-05-05_21-51-54.ckpt'


def  init_variables():
    session.run(tf.global_variables_initializer())

def print_actual_predict():
    msg = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))+"   Evaluated Class: {1}"
    print(msg.format(cnt_pred_cls))


##############################################################################
#CNN
    

print(tf.__version__)

# Variables
# Stop optimization if no improvement found in this many iterations.
require_improvement = 1000
# We know that the images are 8 pixels in each dimension.
img_size = 8
# Images are stored in one-dimensional arrays of the length 64.
img_size_flat = img_size * img_size
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)
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

filter_size3 = 3   #2       # Convolution filters are 3 x 3 pixels.
num_filters3 = 32    #8   # There are 36 of these filters.
#fully-connected layer size --> fc_size klein halten, Exponential features
fc_size = 96 # 128
    
###############################################################################
  

Pixels = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='Pixels')
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
## Eventuell nötig
#conv4 = tf.layers.conv2d(inputs=pool2, name='layer_conv3', padding='same',
#                       filters=num_filters3, kernel_size=filter_size3, activation=tf.nn.relu)
#pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=2, strides=1)
###############################################################################
# flatten layer
flatten = tf.layers.flatten(pool3)
# Fully connected layer
fc1 = tf.layers.dense(inputs=flatten, name='layer_fc1',
                      units=fc_size, activation=tf.nn.relu)
fc2 = tf.layers.dense(inputs=fc1, name='layer_fc_out',
                      units=cnt_classes, activation=None)
###############################################################################
logits = fc2
cnt_pred = tf.nn.softmax(logits=logits)
# compare
cnt_pred_cls = tf.argmax(cnt_pred, axis=1)

restore_path = 'C:/Users/User/switchdrive/HSLU_6_Semester\BAT/projects/Tensorflow/tmp/model_2018-05-14_09-49-38.ckpt'
# Restore variables from disk.
saver.restore(sess=session, save_path=restore_path)


###############################################################################
try:
    while(1):
       feed_dict = {Pixels: image}

        # Calculate the predicted class using TensorFlow.
        cls_pred = session.run(cnt_pred_cls, feed_dict=feed_dict)

        
            
