# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:53:19 2018

First approx Personendetektion low-level tensorflow API

@author: User Daniel Zimmermann
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta
import math
from input_data import read_data_sets
from sklearn.metrics import confusion_matrix
# helpfer fucntion to create weights
def new_weights(shape):
    return tf.Variable(tf.truncate_normal(shape, stddev = 0.05))

# Helper function to create biases
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape =[length]))

def new_conv_layer(input,                    # previous Layer
                  num_input_channels,       # num. channels prev.
                  filter_size,
                  num_filters,              # number filters
                  use_pooling = True):      # use 2x2 pooling downsampling
    # Create shape weights, biases
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape= shape)
    biases = new_biases(length= num_filters)
    # padding macht, dass auch ecken verarbeitet werden
    layer = tf.nn.conv2d(input = input, filter = weights, strides =[1,1,1,1], padding = 'SAME')
    layer += biases
    if use_pooling:      
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1,2,2,1],
                               strides=[1,2,2,1],
                               padding='SAME')
    # max(x,0) non  linearity to the formula
    layer = tf.nn.relu(layer)
    return layer, weights

 # Conversion to fully connected layer   
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    cnt_features = np.array(layer_shape[1:4], dtype=int).prod()
    layer_flat = tf.reshape(layer, [-1, cnt_features])
    return layer_flat, cnt_features

def new_fc_layer(input,
                 cnt_inputs,
                 cnt_outputs,
                 use_relu=True):
    weights = new_weights(shape=[cnt_inputs, cnt_outputs])
    biases = new_biases(length= cnt_outputs)
    layer = tf.matmul(input,weights)+ biases
    
    if use_relu:
        layer = tf.nn.relu(layer)
    
    return layer

def optimize(num_iterations):
    global total_iterations
    start_time = time.time()
    for i in range(total_iterations,
                   total_iterations + num_iterations):
        
        Pixel_batch, cnt_true_batch = data.train.next_batch(batch_size)
        feed_dict_train =  {Pixel: Pixel_batch,
                            cnt_true: cnt_true_batch }      
        session.run(optimizer, feed_dict = feed_dict_train)
        
        if i % 100 == 0:
                acc = session.run(accuracy, feed_dict = feed_dict_train)
                msg ="Optm Iteration {0:>6}, training acc: {1:>6.1%}"
                print(msg.format(i+1,acc))
    total_iterations +=num_iterations
    
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time"+str(timedelta(seconds =int(round(time_dif)))))  

def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(cnt_classes)
    plt.xticks(tick_marks, range(cnt_classes))
    plt.yticks(tick_marks, range(cnt_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show() 
  
# https://www.youtube.com/watch?v=HMcx-zY8JSg


# General Information
learning_rate = 0.001
training_iters = 100
display_step = 10

# Input Value
img_size = 8
img_size_flat = img_size * img_size
cnt_classes = 4
cnt_channels = 1

#Convolutional Layer 1 (Filter)
filter_size1 = 3
cnt_filters1 = 8

#convolutional Layer 2 (Filter)
filter_size2 = 3
cnt_filters2 = 12

#fully-connected layer
fc_size = 20
batch_size = 128

# Import Data 
test = input_data.read_data_sets("/tmp/data/", one_hot=True)

print("Size of:")
print("- Training set")
print ("Test set")


# One Hot encoded, array of 0-12
data.test.cls = np.argmax(data.test.labels, axis = 1)

# Input Layer
Pixel = tf.placeholder(tf.float32, shape =[None, img_size_flat], name='InputPixels')
Pixel_img = tf.reshape(Pixel, [-1, img_size, img_size, cnt_channels])

# Output Layer
cnt_true = tf.placeholder(tf.int32, [None, cnt_classes], name= 'cnt_true')
cnt_true_cls = tf.argmax(cnt_true, dimension=1)


# first layer
layer_conv1, weights_conv1 = \
    new_conv_layer(input = Pixel_img,
                   num_input_channels = cnt_channels,
                   filter_size = filter_size1,
                   num_filters = cnt_filters1,
                   use_pooling= True)

# second layer
layer_conv2, weights_conv2 = \
    new_conv_layer(input = layer_conv1,
                   num_input_channels = cnt_filters1,
                   filter_size = filter_size2,
                   num_filters = cnt_filters2,
                   use_pooling= True)

# input fully connected
layer_flat, cnt_features = flatten_layer(layer_conv2)
layer_fc1 = new_fc_layer(input=layer_flat,
                         cnt_inputs=cnt_features,
                         cnt_outputs=fc_size,
                         use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         cnt_inputs=fc_size,
                         cnt_outputs=cnt_classes,
                         use_relu=False)
 
# prediction
cnt_pred = tf.nn.softmax(layer_fc2)
cnt_pred_cls = tf.argmax(cnt_pred, dimension=1)

# gives advice which 
cross_entropy =tf.nn.sofmax_cross_entropy_with_logits(logits=layer_fc2,
                                                      labels = cnt_true)

# Cost funktion nach Root Mean Square
cost = tf.reduce_mean(cross_entropy)

# Training
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
#optiimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# performance
correct_prediciton = tf.equal(cnt_pred_cls, cnt_true_cls)
# accuracy of 
accuracy = tf.reduce_mean(tf.cast(correct_prediciton, tf.float32))

# needed for Save and Restore evaluated Model params
saver = tf.train.Saver()
# print("Model restored.")

# effective session
session = tf.Session()
session.run(tf.initialize_all_variables)
# Restore variables from disk.
 # saver.restore(sess, "/tmp/model.ckpt")

train_batch_size = 64
total_iterations = 0
# optimize funktion, change the training iterations for better results
optimize(num_iterations=training_iters)

 # Save the variables to disk.
save_path = saver.save(session, "/tmp/model.ckpt")
print("Model saved in path: %s" % save_path)

session.Close()
### eHelpfer function

