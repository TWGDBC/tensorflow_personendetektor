# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:09:56 2018

@author: Daniel Zimmermann
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
import math

# We also need PrettyTensor.
import prettytensor as pt

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)


# We know that MNIST images are 28 pixels in each dimension.
img_size = 8
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)
# Number of colour channels for the images: 1 channel for gray-scale.
cnt_channels = 1
# Number of classes, one class for each of 10 digits.
num_classes = 10
fc_size = 20
batch_size = 128
# lerning rate
learning_rate = 0.001
# number of iterations
num_iterations = 1000
# 
Pixel = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='Pixel')
# batch must be 4 Dimensions, so reshape
Pixel_image = tf.reshape(Pixel, [-1, img_size, img_size, num_channels])
# 
cnt_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='cnt_true')
# compare
cnt_true_cls = tf.argmax(y_true, dimension=1)


# create a layer implementation
net = x_image
# conv layer 1
net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same',
                       filters=16, kernel_size=filter_size, activation=tf.nn.relu)
layer_conv1 = net
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
# conv layer 2
net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=36, kernel_size=filter_size, activation=tf.nn.relu)
layer_conv2 = net
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
# flatten layer
net = tf.contrib.layers.flatten(net)
# Fully connected layer
net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=batch_size, activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=cnt_classes, activation=None)

logits = net
cnt_pred = tf.nn.softmax(logits=logits)
# compare
cnt_true_cls = tf.argmax(cnt_pred, dimension=1)

# loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=cnt_true, logits=logits)
loss = tf.reduce_mean(cross_entropy)

#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

    
weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')



session = tf.Session()

session.run(tf.global_variables_initializer())

# Counter for total number of iterations performed so far.
total_iterations = 0
train_batch_size = 64


     
# creates Tensoflow objects for, return the weights
def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'weights' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.

    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable        
# optimizer equals to Personendetektion V1
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
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
    
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.
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
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
  
    # Split the test-set into smaller batches of this size.
test_batch_size = 256


optimize(num_iterations=num_iterations) # We performed 1000 iterations above.

print_test_accuracy()

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    # Number of images in the test-set.
    num_test = len(data.test.images)
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
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
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