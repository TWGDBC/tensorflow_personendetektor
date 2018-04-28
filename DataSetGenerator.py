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
import numpy as np
import pandas as pd


class data:

    def __init__(self, name, images, labels):
         self.name = name  = ""  
         self.images = images
         self.labels = labels
         self.epochs_completed = 0
         self.index_in_epoch = 0
    
    @property
    def images(self):
        return self.images

    @property
    def labels(self):
        return self.labels

    @property
    def num_examples(self):
        return self.num_examples

    @property
    def epochs_completed(self):
        return self.epochs_completed

    def printName(self):
        print("Name  = " + self.name)

    """Return the next `batch_size` examples from this data set."""
    def next_batch(self, batch_size, shuffle=True):
        start = self.index_in_epoch
        # Shuffle for the first epoch
        if self.epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self.images = self.images[perm0]
            self.labels = self.labels[perm0]
            # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self.num_examples - start
            images_rest_part = self.images[start:self._num_examples]
            labels_rest_part = self.labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self.images = self.images[perm]
                self.labels = self.labels[perm]
                # Start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            images_new_part = self.images[start:end]
            labels_new_part = self.labels[start:end]
            return numpy.concatenate(
                    (images_rest_part, images_new_part), axis=0), numpy.concatenate(
                        (labels_rest_part, labels_new_part), axis=0)
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.images[start:end], self.labels[start:end]
    
            
class train(data):
    name = "train"


    def __init__(self, name):
        self.name = name
        super().__init__(images,labels, geburtsdatum)
 
    def printName(self):
        print("Name  = " + self.name)
        
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
        

    return super().__str__() + " " + self.__personalnummer


class test(data):
    name = "test"

 
    def __init__(self, name):
        self.name = name
        super().__init__(vorname, nachname, geburtsdatum)
 
    def printName(self):
        print("Name  = " + self.name)

    @property
    def images(self):
        return self.images

    @property
    def labels(self):
        return self.labels

    @property
    def num_examples(self):
        return self.num_examples

    @property
    def epochs_completed(self):
       return self._epochs_completed  
       


    
class validation(data):
    name = "validation"

 
    def __init__(self, name):
        self.name = name
 
    def printName(self):
        print("Name  = " + self.name)
        return self._epochs_completed 

             
      
          
dataset_labels = train_dataset[: ,:1]
dataset_images = train_dataset[: ,1:]
print(dataset_labels.shape)
print(dataset_images.shape)            

#Helperfunction getSize of Set
def getTrainSetSize():
    return trainset

#Helperfunction getSize of Set
def setTrainSetSize(row, col, value):
    trainset(row, col) = value
    

#Helperfunction getSize of Set
def DataTo8x8([]):
    
    return 
    

#Helperfunction getSize of Set
def addLabel(LabelValue, arraySize):
    
    new
    return np.concatenate((a, b.T), axis=1)

#Helferfunctions train CSV file
  
    #Helferfunction CheckSame
def CheckSame(filename):
    
    RawData = pd.read_csv("input_data/"+filename).values
    RawData = RawData[0:]
    
for i in range(start, stop = 1):
    
    for j in range(start = 0, stop = 64):
        
            for k in range(start = 0, stop = 64):
                


#loading the data sets from the csv files
print('--------load train & test  & validation file------')

# read training data from CSV file 
print(train_dataset.shape)
train_dataset = train_dataset.as_matrix()

#test dataset
test_dataset = pd.read_csv('../input/test.csv')
print(test_dataset.shape)
test_dataset = test_dataset.as_matrix()

                
## 1 Person
filename1_1 ='_m6_Personen/1p/v2/g_a_output_2018-03-28_19-24-16.csv'
filename1_2 ='_m6_Personen/1p/v2/g_b_output_2018-03-28_19-32-40.csv'
filename1_3 ='_m6_Personen/1p/v2/g_d_output_2018-03-28_19-31-22.csv'
filename1_4 ='_m6_Personen/1p/v2/g_e_output_2018-03-28_19-25-38.csv'
filename1_5 ='_m6_Personen/1p/v2/g_g_output_2018-03-28_19-27-11.csv'
filename1_6 ='_m6_Personen/1p/v2/g_i_output_2018-03-28_19-28-44.csv'
filename1_7 ='_m6_Personen/1p/v2/g_j_output_2018-03-28_19-30-02.csv'
filename1_8 ='_m6_Personen/1p/v1/m_a_output_2018-03-28_18-13-28.csv'
filename1_9 ='_m6_Personen/1p/v1/m_b_output_2018-03-28_18-19-22.csv'
filename1_10 ='_m6_Personen/1p/v1/m_d_output_2018-03-28_18-16-57.csv'
filename1_11 ='_m6_Personen/1p/v1/m_e_output_2018-03-28_18-29-26.csv'
filename1_12 ='_m6_Personen/1p/v1/m_g_output_2018-03-28_18-24-14.csv'
filename1_13 ='_m6_Personen/1p/v1/m_i_output_2018-03-28_18-26-04.csv'
filename1_14 ='_m6_Personen/1p/v1/m_j_output_2018-03-28_18-27-48.csv'
person1_files = [filename1_1, filename1_2, filename1_3, filename1_4, filename1_5, filename1_6, filename1_7, filename1_8, filename1_9, filename1_10,
           filename1_11, filename1_12, filename1_13, filename1_14]

## 2 Persons
filename2_1 ='_m6_Personen/2p/v1/k_a_g_b_output_2018-03-28_18-35-06.csv'
filename2_2 ='_m6_Personen/2p/v1/k_a_g_c_output_2018-03-28_18-32-43.csv'
filename2_3 ='_m6_Personen/2p/v1/k_a_g_i_output_2018-03-28_18-42-08.csv'
filename2_4 ='_m6_Personen/2p/v1/k_d_g_e_output_2018-03-28_18-38-29.csv'
filename2_5 ='_m6_Personen/2p/v1/k_d_g_f_output_2018-03-28_18-36-53.csv'
filename2_6 ='_m6_Personen/2p/v1/k_g_g_c_output_2018-03-28_18-40-21.csv'
filename2_7 ='_m6_Personen/2p/v1/k_j_g_e_output_2018-03-28_18-43-34.csv'
filename2_8 ='_m6_Personen/2p/v1/k_j_g_i_output_2018-03-28_18-45-04.csv'
filename2_9 ='_m6_Personen/2p/v2/a_b_output_2018-03-28_19-36-43.csv'
filename2_10 ='_m6_Personen/2p/v2/a_c_output_2018-03-28_19-35-19.csv'
filename2_11 ='_m6_Personen/2p/v2/a_i_output_2018-03-28_19-42-19.csv'
filename2_12 ='_m6_Personen/2p/v2/d_e_output_2018-03-28_19-39-32.csv'
filename2_12 ='_m6_Personen/2p/v2/d_f_output_2018-03-28_19-38-07.csv'
filename2_13 ='_m6_Personen/2p/v2/g_c_output_2018-03-28_19-40-57.csv'
filename2_14='_m6_Personen/2p/v2/j_e_output_2018-03-28_19-43-38.csv'
filename2_15='_m6_Personen/2p/v2/j_i_output_2018-03-28_19-45-03.csv'
person2_files = [filename2_1, filename2_2, filename2_3, filename2_4, filename2_5, filename2_6, filename2_7, filename2_8, filename2_9, filename2_10,
                 filename2_11, filename2_12, filename2_13, filename2_14, filename2_15]

## 3 Persons
filename3_1 ='_m6_Personen/3p/v1/k_a_m_b_k_c_output_2018-03-28_18-47-19.csv'
filename3_2 ='_m6_Personen/3p/v1/k_a_m_e_k_i_output_2018-03-28_18-52-43.csv'
filename3_3 ='_m6_Personen/3p/v1/k_d_m_e_k_f_output_2018-03-28_18-49-21.csv'
filename3_4 ='_m6_Personen/3p/v1/k_g_m_e_k_i_output_2018-03-28_18-54-39.csv'
filename3_5 ='_m6_Personen/3p/v1/k_a_g_e_k_i_output_2018-03-28_19-18-52.csv'
filename3_6 ='_m6_Personen/3p/v1/k_j_m_h_k_l_output_2018-03-28_18-56-51.csv'
filename3_7 ='_m6_Personen/3p/v1/k_g_m_k_k_i_output_2018-03-28_18-58-23.csv'
filename3_8 ='_m6_Personen/3p/v1/k_h_g_d_k_loutput_2018-03-28_19-20-56.csv'
person3_files = [filename3_1, filename3_2, filename3_3, filename3_4, filename3_5, filename3_6, filename3_7, filename3_8]

## 4 Personen
filename4_1 ='_m6_Personen/4p/v1/p4_acdf_output_2018-03-28_19-02-54.csv'
filename4_2 ='_m6_Personen/4p/v1/p4_gjcf_output_2018-03-28_19-09-47.csv'
filename4_3 ='_m6_Personen/4p/v1/p4_acjl_output_2018-03-28_19-00-55.csv'
filename4_4 ='_m6_Personen/4p/v1/p4_acdf_output_2018-03-28_19-04-29.csv'
filename4_5 ='_m6_Personen/4p/v1/p4_dfgi_output_2018-03-28_19-06-25.csv'
filename4_6 ='_m6_Personen/4p/v1/p4_adgj_output_2018-03-28_19-08-05.csv'
person4_files = [filename4_1, filename4_2, filename4_3, filename4_4, filename4_5, filename4_6]


print("------finish loading --------------------")





np.savetxt('trainset.csv', np.c_[range(1,len(trainSet)+1),], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
        


