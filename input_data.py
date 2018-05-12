# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 00:56:16 2018

@author: User
"""
# pyhton 2.7 ability
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Import classes and functions
import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np
import pandas as pd


CSV_COLUMN_NAMES = ['Pixel 0' ,'Pixel 1' ,'Pixel 2' ,'Pixel 3' ,'Pixel 4' ,'Pixel 5' ,'Pixel 6' ,'Pixel 7' ,'Pixel 8' ,
'Pixel 9' ,'Pixel 10' ,'Pixel 11' ,'Pixel 12' ,'Pixel 13' ,'Pixel 14' ,'Pixel 15' ,'Pixel 16' ,
'Pixel 17' ,'Pixel 18' ,'Pixel 19' ,'Pixel 20' ,'Pixel 21' ,'Pixel 22' ,'Pixel 23' ,'Pixel 24' ,
'Pixel 25' ,'Pixel 26' ,'Pixel 27' ,'Pixel 28' ,'Pixel 29' ,'Pixel 30' ,'Pixel 31' ,'Pixel 32' ,
'Pixel 33' ,'Pixel 34' ,'Pixel 35' ,'Pixel 36' ,'Pixel 37' ,'Pixel 38' ,'Pixel 39' ,'Pixel 40' ,
'Pixel 41' ,'Pixel 42' ,'Pixel 43' ,'Pixel 44' ,'Pixel 45' ,'Pixel 46' ,'Pixel 47' ,'Pixel 48' ,
'Pixel 49' ,'Pixel 50' ,'Pixel 51' ,'Pixel 52' ,'Pixel 53' ,'Pixel 54' ,'Pixel 55' ,'Pixel 56' , 
'Pixel 57' ,'Pixel 58' ,'Pixel 59' ,'Pixel 60' ,'Pixel 61' ,'Pixel 62' ,'Pixel 63']


def extract_images(f):

    data = pd.read_csv(f, names=CSV_COLUMN_NAMES, sep=',',header=0)
    data = data.as_matrix()
    return data

def extract_labels(f, one_hot, num_classes=5):
    
   labels = pd.read_csv(f, sep=',', header=0)
   labels = labels.as_matrix()
   if (one_hot == True):
       num_labels = len(labels)
       index_offset = np.arange(num_labels) * num_classes
       labels_one_hot = np.zeros((num_labels, num_classes))
       labels_one_hot.flat[index_offset + labels.ravel()] = 1
       return labels_one_hot
   return labels

class DataSet(object):

    def __init__(self, name, images, labels, reshape = False):
        
         assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
         
         self._num_examples = images.shape[0]
         self._name = " "  
         self._images = images
         self._labels = labels
         self._epochs_completed = 0
         self._index_in_epoch = 0
    
    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def printName(self):
        print("Name  = " + self.name)

    """Return the next `batch_size` examples from this data set."""
    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
            # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
                # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate(
                    (images_rest_part, images_new_part), axis=0), np.concatenate(
                        (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]
    
    
def read_data_sets(train_dir,
                   one_hot=True,
                   reshape=True,
                   shuffle=False,
                   validation_size=20000):
    #TEST_IMAGES = 'testImages_2018-04-30_14-51-14.csv'      # Profil 0
    #TEST_IMAGES = 'testImages_2018-05-08_15-01-33.csv'      # Profil 1
    #TEST_IMAGES = 'testImages_2018-05-11_16-59-49.csv'      # Profil 2

    
    #TEST_LABELS = 'testLabels_2018-04-30_14-51-17.csv'     # Profil 0
    #TEST_LABELS = 'testLabels_2018-05-08_15-01-48.csv'     # Profil 1
    #TEST_LABELS = 'testLabels_2018-05-11_16-59-55.csv'      # Profil 2

    TEST_IMAGES ='trainImages_2018-04-30_14-51-15.csv'    # Profil 0
    #TRAIN_IMAGES ='trainImages_2018-05-08_15-01-34.csv'    # Profil 1 generisch
    #TRAIN_IMAGES ='trainImages_2018-05-11_16-59-51.csv'    # Profil 2 rein
    TRAIN_IMAGES = 'trainImages_2018-05-12_08-26-45.csv'    # Profi 2 generisch
    
   
    TEST_LABELS ='trainLabels_2018-04-30_14-51-17.csv'    # Profil 0 
    #TRAIN_LABELS ='trainLabels_2018-05-08_15-01-48.csv'    # Profil 1 generisch
    #TRAIN_LABELS ='trainLabels_2018-05-11_16-59-55.csv'    # Profil 2 rein
    TRAIN_LABELS ='trainLabels_2018-05-12_08-27-00.csv'    #Profil 2 generisch
    

    
    train_labels = extract_labels(train_dir+TRAIN_LABELS, one_hot=one_hot)
    train_images = extract_images(train_dir+TRAIN_IMAGES)
    test_labels = extract_labels(train_dir+TEST_LABELS, one_hot=one_hot)  
    test_images = extract_images(train_dir+TEST_IMAGES)
    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
                'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_images), validation_size))
    if shuffle:
        perm0 = np.arange(len(train_labels))
        np.random.shuffle(perm0)
        train_images = train_images[perm0]
        train_labels = train_labels[perm0]      
    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]
        

    train = DataSet(name = 'train', images = train_images, labels= train_labels)
    validation = DataSet(name = 'valid', images = validation_images, labels= validation_labels)
    test = DataSet(name = 'test', images = test_images, labels = test_labels)

    return train, test, validation

    #return train_images, train_labels, test_images, test_labels, validation_images, validation_labels






 
 

