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
import numpy as np
from datetime import timedelta
from sklearn.metrics import confusion_matrix
import datetime
import matplotlib.pyplot as plt
import math
import time
import input_data

print(tf.__version__)

# ATTENTITON Projectpath must be declared
projectpath = 'C:/Users/User/switchdrive/HSLU_6_Semester/BAT/projects/Tensorflow/tmp/'
# to restore an created model use this line and uncomment
restore_path = projectpath+'model_2018-05-25_00-03-05.ckpt'
showRestore = True;
showOptimize = False;
#showRestore = False;
#showOptimize = True;


###############################################################################
# changeable Variables
###############################################################################
# defines if the dataset should be shuffled first
shuffle = True
# learning rate
learning_rate = 0.001
# number of iterations
num_iterations = 10000
# batch sizes
validation_size = 25000
train_batch_size = 500 # batch size is ~1% of the training size
test_batch_size = 50   # batch size is ~1 % of the test size
# Stop optimization if no improvement found in this many iterations.
require_improvement = 1000
###############################################################################
# Convolutional Layer 1.
filter_size1 = 6      # Convolution filters are 5 x 5 pixels.
num_filters1 = 8     # There are 16 of these filters.
strides1 = 1
poolsize1 = 2
# Convolutional Layer 2.
filter_size2 = 3      # Convolution filters are 3 x 3 pixels.
num_filters2 = 16     # There are 16 of these filters.
strides2 = 1
poolsize2 = 2
# Convolutional Layer 3.
filter_size3 = 3      # Convolution filters are 3 x 3 pixels.
num_filters3 = 16     # There are 32 of these filters.
strides3 = 1
poolsize3 = 3
# Convolutional Layer 4.
filter_size4 = 3      # Convolution filters are 3 x 3 pixels.
num_filters4 = 32     # There are 32 of these filters.
strides4 = 1
poolsize4 = 2
#fully-connected layer size --> fc_size klein halten, Exponential features
fc_size = 54
###############################################################################
# Helper Variables
###############################################################################
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
# Best validation accuracy seen so far.# Best v 
best_validation_accuracy = 0.0
# Iteration-number for last improvement to validation accuracy.
last_improvement = 0
# Counter for total number of iterations performed so far.
total_iterations = 0
# batch size to lower RAM consum
batch_size = 500  

# creates Tensoflow objects for, return the weights
def init_variables():
    session.run(tf.global_variables_initializer())
    
 
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.
    # Get the true classifications for the test-set.
    cls_true = test.cls 
    labels = [0,1,2,3,4]
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred, labels= labels)
# hier ansetzen
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

# plots teh conv_weights of the     
def plot_conv_weights(weights, input_channel=0):

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Print mean and standard deviation.
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show() 


    
# optimizer best improvements
def optimize(num_iterations):
    # Ensure we update the global variables rather than local copies.
    global total_iterations
    global best_validation_accuracy
    global last_improvement

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):

        # Increase the total number of iterations performed.
        total_iterations += 1

        # Get a batch of training examples.
        # cnt_batch now holds a batch of images and
        # cnt_true_batch are the true labels for those images.
        cnt_batch, cnt_true_batch = train.next_batch(train_batch_size, shuffle = shuffle)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {Pixels: cnt_batch,
                           cnt_true: cnt_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations and after last iteration.
        if (total_iterations % 100 == 0) or (i == (num_iterations - 1)):

            # Calculate the accuracy on the training-batch.
            acc_train = session.run(accuracy, feed_dict=feed_dict_train)

            # Calculate the accuracy on the validation-set.
            # The function returns 2 values but we only need the first.
            acc_validation, _ = validation_accuracy()

            # If validation accuracy is an improvement over best-known.
            if acc_validation > best_validation_accuracy:
                # Update the best-known validation accuracy.
                best_validation_accuracy = acc_validation
                
                # Set the iteration for the last improvement to current.
                last_improvement = total_iterations

                # Save all variables of the TensorFlow graph to file.
                saver.save(sess=session, save_path=save_path)

                # A string to be printed below, shows improvement found.
                improved_str = '*'
            else:
                # An empty string to be printed below.
                # Shows that no improvement was found.
                improved_str = ''
            
            # Status-message for printing.
            msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.3%}, Validation Acc: {2:>6.3%} {3}"

            # Print it.
            print(msg.format(i + 1, acc_train, acc_validation, improved_str))

        # If no improvement found in the required number of iterations.
        if total_iterations - last_improvement > require_improvement:
            print("No improvement found in a while, stopping optimization.")

            # Break out from the for-loop.
            break

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def print_test_accuracy(show_confusion_matrix=True):

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
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {Pixels: images[i:j, :],
                     cnt_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(cnt_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    
    correct = (cls_true == cls_pred)

    return correct, cls_pred

def  predict_cls_testpredict_ ():
    return predict_cls(images = test.images,
                       labels = test.labels,
                       cls_true = test.cls)


def predict_cls_validation():
    return predict_cls(images = validation.images,
                       labels = validation.labels,
                       cls_true = validation.cls)    
    

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
    
  

###############################################################################
# Load training and test data
# prepared Train Test Data, Validation Data is shuffle from Train
(train, test, validation) = input_data.read_data_sets('prepared_data/', validation_size = validation_size, shuffle=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(train.images)))
print("- Test-set:\t\t{}".format(len(test.images)))
print("- Validation-set:\t{}".format(len(validation.images)))
###############################################################################

tf.reset_default_graph() 

test.cls = np.argmax(test.labels, axis=1)
validation.cls = np.argmax(validation.labels, axis=1)


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
#pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=poolsize1, strides=strides1)
###############################################################################
# conv layer 2
conv2 = tf.layers.conv2d(inputs=conv1, name='layer_conv2', padding='same',
                       filters=num_filters2, kernel_size=filter_size2, activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=poolsize2, strides=strides2)
###############################################################################
## conv layer 3
conv3 = tf.layers.conv2d(inputs=pool2, name='layer_conv3', padding='same',
                       filters=num_filters3, kernel_size=filter_size3, activation=tf.nn.relu)
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=poolsize3, strides=strides3)
###############################################################################
## conv layer 4
conv4 = tf.layers.conv2d(inputs=pool3, name='layer_conv4', padding='same',
                       filters=num_filters4, kernel_size=filter_size4, activation=tf.nn.relu)
#pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=poolsize4, strides=strides4)
###############################################################################
# flatten layer
flatten = tf.layers.flatten(conv3)
# Fully connected layer
fc1 = tf.layers.dense(inputs=flatten, name='layer_fc1',
                      units=fc_size, activation=tf.nn.relu)
fc2 = tf.layers.dense(inputs=fc1, name='layer_fc_out',
                      units=cnt_classes, activation=None)

logits = fc2
cnt_pred = tf.nn.softmax(logits=logits)
# compare
cnt_pred_cls = tf.argmax(cnt_pred, axis=1)

# loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=cnt_true, logits=logits)
loss = tf.reduce_mean(cross_entropy, name= "loss")
    
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


## Alle nicht optimal für lösung
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
#optimizer = tf.train.AdadeltaOptimizer(learning_rate = learning_rate).minimize(loss)
#optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate).minimize(loss)
#optimizer = tf.train.AdagradOptimizer(learning_rate = learning_rate).minimize(loss)

correct_prediction  = tf.equal(cnt_pred_cls, cnt_true_cls, name = 'correct_prediciton')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

# needed for Save and Restore evaluated Model params
saver = tf.train.Saver(max_to_keep= 200)
save_path = projectpath+'model_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))+'.ckpt'


# graphical 
#writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

tf.summary.scalar("loss", loss)
tf.summary.histogram("histogram_loss", loss)
summary_op = tf.summary.merge_all()



session = tf.Session()

writer = tf.summary.FileWriter('./graphs', session.graph)

init_variables()

if showOptimize:
    
    optimize(num_iterations=num_iterations) # We performed 1000 iterations above.
    print_test_accuracy(True)

#Restore variables from disk.

if showRestore:
    saver.restore(sess=session, save_path=restore_path)
    print_test_accuracy(True)

