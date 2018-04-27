# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:02:12 2018

@author: User
"""

import tensorflow as tf

from tensorflow.contrib.data import Dataset, Iterator

# Train Sets Pixel 1 - Pixel 64  max 4 Personen
train_datasets_p0 = tf.constant(['train/img1.png',
                              'train/img2.png'
                              'train/img3.png',
                              'train/img4.png',
                              'train/img5.png',
                              'train/img6.png'])
train_datasets_p1 = tf.constant(['train/img1.png',
                              'train/img2.png'
                              'train/img3.png',
                              'train/img4.png',
                              'train/img5.png',
                              'train/img6.png'])
train_datasets_p2 = tf.constant(['train/img1.png',
                              'train/img2.png'
                              'train/img3.png',
                              'train/img4.png',
                              'train/img5.png',
                              'train/img6.png'])
train_datasets_p3 = tf.constant(['train/img1.png',
                              'train/img2.png'
                              'train/img3.png',
                              'train/img4.png',
                              'train/img5.png',
                              'train/img6.png'])
train_datasets_p4 = tf.constant(['train/img1.png',
                              'train/img2.png'
                              'train/img3.png',
                              'train/img4.png',
                              'train/img5.png',
                              'train/img6.png'])

train_datasets_all = [train_datasets_p0, train_datasets_p1, train_datasets_p2, train_datasets_p3, train_datasets_p4]


train_labels = tf.constant([0, 1, 2, 3, 4])
val_labels = tf.constant([0, 1, 2, 3, 4])

val_imgs = tf.constant(['val/img1.png',
                        'val/img2.png',
                        'val/img3.png',
                        'val/img4.png'])


# Validation Sets Pixel 1 - Pixel 64  max 4 Personen


# create TensorFlow Dataset objects
tr_data = Dataset.from_tensor_slices((, train_labels))
val_data = Dataset.from_tensor_slices((val_imgs, val_labels))

# create TensorFlow Iterator object
iterator = Iterator.from_structure(tr_data.output_types,
                                   tr_data.output_shapes)
next_element = iterator.get_next()

# create two initialization ops to switch between the datasets
training_init_op = iterator.make_initializer(tr_data)
validation_init_op = iterator.make_initializer(val_data)

with tf.Session() as sess:

    # initialize the iterator on the training data
    sess.run(training_init_op)

    # get each element of the training dataset until the end is reached
    while True:
        try:
            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break

    # initialize the iterator on the validation data
    sess.run(validation_init_op)

    # get each element of the validation dataset until the end is reached
    while True:
        try:
            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break