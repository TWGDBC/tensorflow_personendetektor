# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:10:06 2018

@author: User
"""
# pyhton 2.7 ability
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import classes and functions
import tensorflow as tf
#import prettytensor as pt
import numpy as np
from datetime import timedelta
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import math
import time
import input_data

tf.__version__

# Variables

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
# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.
# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.
#fully-connected layer size
fc_size = 64


def  init_variables():
    session.run(tf.global_variables_initializer())
    
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)
        # Get the images from the test-set between index i and j.
        images = test.images[i:j, :]
        # Get the associated labels.
        labels = test.labels[i:j, :]
        # Create a feed-dict with these images and labels.
        feed_dict = {Pixels: images,
                     cnt_true: labels}
        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(cnt_pred_cls, feed_dict=feed_dict)
        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j
    # Convenience variable for the true class-numbers of the test-set.
    cls_true = test.cls 
    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()
    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test
    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
       # plot_example_errors(cls_pred=cls_pred, correct=correct)

def predict_cls(images, labels, cls_true):

        feed_dict = {Pixels: images :],
                     cnt_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred = session.run(cnt_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    
    correct = (cls_true == cls_pred)

    return correct, cls_pred

    

def cls_accuracy(correct):
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / len(correct)

    return acc, correct_sum

def validation_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    # The function returns two values but we only need the first.
    correct, _ = predict_cls_validation()
      # Calculate the classification accuracy and return it.
    return cls_accuracy(correct)
    


tf.reset_default_graph() 

Pixels = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='Pixels')
# batch must be 4 Dimensions, so reshape
#Pixel_mat = tf.reshape(Pixels, [-1 ,img_size, img_size], name='Pixel_mat')
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
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)
# conv layer 2
conv2 = tf.layers.conv2d(inputs=pool1, name='layer_conv2', padding='same',
                       filters=num_filters2, kernel_size=filter_size2, activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=1)
# flatten layer
pool2 = tf.layers.flatten(pool2)
# Fully connected layer
fc1 = tf.layers.dense(inputs=pool2, name='layer_fc1',
                      units=fc_size, activation=tf.nn.relu)
fc2 = tf.layers.dense(inputs=fc1, name='layer_fc_out',
                      units=cnt_classes, activation=None)

logits = fc2
cnt_pred = tf.nn.softmax(logits=logits)
# compare
cnt_pred_cls = tf.argmax(cnt_pred, dimension=1)

# loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=cnt_true, logits=logits)
loss = tf.reduce_mean(cross_entropy, name= "loss")
    
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

    
# needed for Save and Restore evaluated Model params
saver = tf.train.Saver(max_to_keep= 100)
save_path = 'C:/Users/User/switchdrive/HSLU_6_Semester/BAT/projects/tensorflow_personendetektor/tmp/model_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))+'.ckpt'
restore_path = 'C:/Users/User/switchdrive/HSLU_6_Semester/BAT/projects/tensorflow_personendetektor/tmp/model_2018-05-05_21-51-54.ckpt'

tf.summary.scalar("loss", loss)
tf.summary.histogram("histogram_loss", loss)
summary_op = tf.summary.merge_all()

session = tf.Session()

writer = tf.summary.FileWriter('./graphs', session.graph)

init_variables()
#print_test_accuracy()
optimize(num_iterations=num_iterations) # We performed 1000 iterations above.

print_test_accuracy()

# Restore variables from disk.

try:
saver.restore(sess=session, save_path=restore_path)

print_test_accuracy()

