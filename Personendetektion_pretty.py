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
import prettytensor as pt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import timedelta
import time
import math
# helper functions
    
# creates Tensoflow objects for, return the weights
def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'weights' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.

    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable   
     
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
        # It is easier to update it in each iteration because
        # we need this number several times in the following.
        total_iterations += 1

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: cnt_batch,
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
            msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"

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

# plots the incorrect images
def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = test.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
    
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = test.cls
    
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
        images = data.test.images[i:j, :]
        # Get the associated labels.
        labels = data.test.labels[i:j, :]
        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}
        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(cnt_pred_cls, feed_dict=feed_dict)
        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j
    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls
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
        plot_example_errors(cls_pred=cls_pred, correct=correct)
    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


# Load the training data into two NumPy arrays
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

train_features_placeholder = tf.placeholder(features.dtype, features.shape)
train_labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

trainSet = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))


print("Size of:")
print("- Training-set:\t\t{}".format(len(train.labels)))
print("- Test-set:\t\t{}".format(len(test.labels)))
print("- Validation-set:\t{}".format(len(validation.labels)))

test.cls = np.argmax(test.labels, axis=1)

# changes teh All train set data in a 8x8 image as flaot 32 -> also available on raspberry pi
train.images = trainSet[:, 1:].reshape(train.shape[0], 1, 8, 8).astype( 'float32' )
test = testSet.reshape(test.shape[0], 1, 8, 8).astype( 'float32' )


# We know that MNIST images are 28 pixels in each dimension.
img_size = 8
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)
# Number of colour channels for the images: 1 channel for gray-scale.
cnt_channels = 1
# Number of classes, one class for each of 10 digits.
cnt_classes = 5
#fully-connected layer size
fc_size = 50


# lerning rate
learning_rate = 0.001
# number of iterations
num_iterations = 1000
# 
Pixel = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='Pixel')
# batch must be 4 Dimensions, so reshape
Pixel_image = tf.reshape(Pixel, [-1, img_size, img_size, cnt_channels])
# 
cnt_true = tf.placeholder(tf.float32, shape=[None, cnt_classes], name='cnt_true')
# compare
cnt_true_cls = tf.argmax(cnt_true, dimension=1)

# create objects of xpretty form the image
if False:
    cnt_cnn = pt.wrap(Pixel_image)

# pretty Tensor makes the hole layers easier 
# pt.defualts_scope is automatic relu, else activation_fn = 
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        cnt_pred, loss = cnt_cnn.\
            conv2d(kernel=5, depth=16, name='layer_conv1').\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=2, depth=16, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=fc_size, name='layer_fc1').\
            softmax_classifier(num_classes=cnt_classes, labels=cnt_true)
    
    
weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')
    
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

# needed for Save and Restore evaluated Model params
saver = tf.train.Saver(max_to_keep= 100)
save_path = '/tmp/model.ckpt'

# graphical 
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

tf.summary.scalar("loss", loss)
tf.summary.histogram("histogram loss", loss)
summary_op = tf.summary.merge_all()

session = tf.Session()
# Restore variables from disk.
 # saver.restore(session, "/tmp/model.ckpt")
 
session.run(tf._initialize_all_variables())
# writer = tf.summary.FileWriter('./graphs', sess.graph)
# Counter for total number of iterations performed so far.
total_iterations = 0
train_batch_size = 64
# Best validation accuracy seen so far.
best_validation_accuracy = 0.0
# Iteration-number for last improvement to validation accuracy.
last_improvement = 0
# Stop optimization if no improvement found in this many iterations.
require_improvement = 1000


optimize(num_iterations=num_iterations) # We performed 1000 iterations above.

print_test_accuracy()



 