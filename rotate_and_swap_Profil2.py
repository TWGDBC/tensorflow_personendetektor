# -*- coding: utf-8 -*-
"""
Created on Tue April  18 09:35:35 2018

Skript to Rotate and Swap defined Data, to create more Datasets
Profil 2
Creates data in /input_data/...


@author: Daniel Zimmermann
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd


#configure rotate and swip you want to execute
swap = True
rotate90 = True
rotate180 = True
rotate270 = True
###############################################################################

#Helpfer functions  

"""
swapfkt:    Changes the CSV File columns, that the 8x8 Pixel Matrix is swapped
param:  filelist, list of strings, which gives paths to the swapable CSV-File
return: -

"""  
def swapfkt(filelist):
    for value in filelist: 
        data = pd.read_csv(value, names=CSV_COLUMN_NAMES, header=1)
        DataSwap = pd.DataFrame()
        DataSwap = data[['Pixel 7' ,'Pixel 6' ,'Pixel 5' ,'Pixel 4' ,'Pixel 3' ,'Pixel 2' ,'Pixel 1' ,'Pixel 0',
                       'Pixel 15' ,'Pixel 14' ,'Pixel 13' ,'Pixel 12' ,'Pixel 11' ,'Pixel 10' ,'Pixel 9' ,'Pixel 8',
                       'Pixel 23' ,'Pixel 22' ,'Pixel 21' ,'Pixel 20' ,'Pixel 19' ,'Pixel 18' ,'Pixel 17' ,'Pixel 16',
                       'Pixel 31' ,'Pixel 30' ,'Pixel 29' ,'Pixel 28' ,'Pixel 27' ,'Pixel 26' ,'Pixel 25' ,'Pixel 24',
                       'Pixel 39' ,'Pixel 38' ,'Pixel 37' ,'Pixel 36' ,'Pixel 35' ,'Pixel 34' ,'Pixel 33' ,'Pixel 32',
                       'Pixel 47' ,'Pixel 46' ,'Pixel 45' ,'Pixel 44' ,'Pixel 43' ,'Pixel 42' ,'Pixel 41' ,'Pixel 40',
                       'Pixel 55' ,'Pixel 54'  ,'Pixel 53' ,'Pixel 52' ,'Pixel 51' ,'Pixel 50' ,'Pixel 49' ,'Pixel 48',
                       'Pixel 63' ,'Pixel 62'  ,'Pixel 61' ,'Pixel 60' ,'Pixel 59' ,'Pixel 58' ,'Pixel 57' ,'Pixel 56',
                       'Thermistor' ,'Timestamp']]    
        DataSwap.to_csv(value[:-4]+'_swap.csv',
                    sep=',',
                    encoding='utf-8',
                    index=False)
 
"""
swapfkt:    Changes the CSV File columns, that the 8x8 Pixel Matrix is rotated 90 degree
param:  filelist, list of strings, which gives paths to the rotateable CSV-File
return: -

""" 
def rotate90fkt(filelist):
    for value in filelist: 
          data = pd.read_csv(value, names=CSV_COLUMN_NAMES, header=1)
          Data90 = pd.DataFrame()
          Data90 = data[['Pixel 7' ,'Pixel 15' ,'Pixel 23' ,'Pixel 31' ,'Pixel 39' ,'Pixel 47' ,'Pixel 55' ,'Pixel 63',
                         'Pixel 6' ,'Pixel 14' ,'Pixel 22' ,'Pixel 30' ,'Pixel 38' ,'Pixel 46' ,'Pixel 54' ,'Pixel 62',
                         'Pixel 5' ,'Pixel 13' ,'Pixel 21' ,'Pixel 29' ,'Pixel 37' ,'Pixel 45' ,'Pixel 53' ,'Pixel 61',
                         'Pixel 4' ,'Pixel 12' ,'Pixel 20' ,'Pixel 28' ,'Pixel 36' ,'Pixel 44' ,'Pixel 52' ,'Pixel 60',
                         'Pixel 3' ,'Pixel 11' ,'Pixel 19' ,'Pixel 27' ,'Pixel 35' ,'Pixel 43' ,'Pixel 51' ,'Pixel 59',
                         'Pixel 2' ,'Pixel 10' ,'Pixel 18' ,'Pixel 26' ,'Pixel 34' ,'Pixel 42' ,'Pixel 50' ,'Pixel 58',
                         'Pixel 1' ,'Pixel 9'  ,'Pixel 17' ,'Pixel 25' ,'Pixel 33' ,'Pixel 41' ,'Pixel 49' ,'Pixel 57',
                         'Pixel 0' ,'Pixel 8'  ,'Pixel 16' ,'Pixel 24' ,'Pixel 32' ,'Pixel 40' ,'Pixel 48' ,'Pixel 56',
                         'Thermistor' ,'Timestamp']]
          Data90.to_csv(value[:-4]+'_rotate90.csv',
                        sep=',',
                        encoding='utf-8',
                        index=False)

"""
swapfkt:    Changes the CSV File columns, that the 8x8 Pixel Matrix is rotated 180 degree
param:  filelist, list of strings, which gives paths to the rotateable CSV-File
return: -

""" 
def rotate180fkt(filelist):  
   for value in filelist: 
          data = pd.read_csv(value, names=CSV_COLUMN_NAMES, header=1)
          Data180 = pd.DataFrame()
          Data180 = data[['Pixel 63' ,'Pixel 62' ,'Pixel 61' ,'Pixel 60' ,'Pixel 59' ,'Pixel 58' ,'Pixel 57' ,'Pixel 56',
                         'Pixel 55' ,'Pixel 54' ,'Pixel 53' ,'Pixel 52' ,'Pixel 51' ,'Pixel 50' ,'Pixel 49' ,'Pixel 48',
                         'Pixel 47' ,'Pixel 46' ,'Pixel 45' ,'Pixel 44' ,'Pixel 43' ,'Pixel 42' ,'Pixel 41' ,'Pixel 40',
                         'Pixel 39' ,'Pixel 38' ,'Pixel 37' ,'Pixel 36' ,'Pixel 35' ,'Pixel 34' ,'Pixel 33' ,'Pixel 32',
                         'Pixel 31' ,'Pixel 30' ,'Pixel 29' ,'Pixel 28' ,'Pixel 27' ,'Pixel 26' ,'Pixel 25' ,'Pixel 24',
                         'Pixel 23' ,'Pixel 22' ,'Pixel 21' ,'Pixel 20' ,'Pixel 19' ,'Pixel 18' ,'Pixel 17' ,'Pixel 16',
                         'Pixel 15' ,'Pixel 14'  ,'Pixel 13' ,'Pixel 12' ,'Pixel 11' ,'Pixel 10' ,'Pixel 9' ,'Pixel 8',
                         'Pixel 7' ,'Pixel 6'  ,'Pixel 5' ,'Pixel 4' ,'Pixel 3' ,'Pixel 2' ,'Pixel 1' ,'Pixel 0',
                         'Thermistor' ,'Timestamp']]
          Data180.to_csv(value[:-4]+'_rotate180.csv',
                        sep=',',
                        encoding='utf-8',
                        index=False)

"""
swapfkt:    Changes the CSV File columns, that the 8x8 Pixel Matrix is rotated 270 degree
param:  filelist, list of strings, which gives paths to the rotateable CSV-File
return: -

"""           
def rotate270fkt(filelist):  
   for value in filelist: 
          data = pd.read_csv(value, names=CSV_COLUMN_NAMES, header=1)
          Data270 = pd.DataFrame()
          Data270 = data[['Pixel 56' ,'Pixel 48' ,'Pixel 40' ,'Pixel 32' ,'Pixel 24' ,'Pixel 16' ,'Pixel 8' ,'Pixel 0',
                         'Pixel 57' ,'Pixel 49' ,'Pixel 41' ,'Pixel 33' ,'Pixel 25' ,'Pixel 17' ,'Pixel 9' ,'Pixel 1',
                         'Pixel 58' ,'Pixel 50' ,'Pixel 42' ,'Pixel 34' ,'Pixel 26' ,'Pixel 18' ,'Pixel 10' ,'Pixel 2',
                         'Pixel 59' ,'Pixel 51' ,'Pixel 43' ,'Pixel 35' ,'Pixel 27' ,'Pixel 19' ,'Pixel 11' ,'Pixel 3',
                         'Pixel 60' ,'Pixel 52' ,'Pixel 44' ,'Pixel 36' ,'Pixel 28' ,'Pixel 20' ,'Pixel 12' ,'Pixel 4',
                         'Pixel 61' ,'Pixel 53' ,'Pixel 45' ,'Pixel 37' ,'Pixel 29' ,'Pixel 21' ,'Pixel 13' ,'Pixel 5',
                         'Pixel 62' ,'Pixel 54' ,'Pixel 46' ,'Pixel 38' ,'Pixel 30' ,'Pixel 22' ,'Pixel 14' ,'Pixel 6',
                         'Pixel 63' ,'Pixel 55' ,'Pixel 47' ,'Pixel 39' ,'Pixel 31' ,'Pixel 23' ,'Pixel 15' ,'Pixel 7',
                         'Thermistor' ,'Timestamp']]
          Data270.to_csv(value[:-4]+'_rotate270.csv',
                        sep=',',
                        encoding='utf-8',
                        index=False)
###############################################################################
CSV_COLUMN_NAMES = ['Pixel 0' ,'Pixel 1' ,'Pixel 2' ,'Pixel 3' ,'Pixel 4' ,'Pixel 5' ,'Pixel 6' ,'Pixel 7' ,'Pixel 8' ,
'Pixel 9' ,'Pixel 10' ,'Pixel 11' ,'Pixel 12' ,'Pixel 13' ,'Pixel 14' ,'Pixel 15' ,'Pixel 16' ,
'Pixel 17' ,'Pixel 18' ,'Pixel 19' ,'Pixel 20' ,'Pixel 21' ,'Pixel 22' ,'Pixel 23' ,'Pixel 24' ,
'Pixel 25' ,'Pixel 26' ,'Pixel 27' ,'Pixel 28' ,'Pixel 29' ,'Pixel 30' ,'Pixel 31' ,'Pixel 32' ,
'Pixel 33' ,'Pixel 34' ,'Pixel 35' ,'Pixel 36' ,'Pixel 37' ,'Pixel 38' ,'Pixel 39' ,'Pixel 40' ,
'Pixel 41' ,'Pixel 42' ,'Pixel 43' ,'Pixel 44' ,'Pixel 45' ,'Pixel 46' ,'Pixel 47' ,'Pixel 48' ,
'Pixel 49' ,'Pixel 50' ,'Pixel 51' ,'Pixel 52' ,'Pixel 53' ,'Pixel 54' ,'Pixel 55' ,'Pixel 56' , 
'Pixel 57' ,'Pixel 58' ,'Pixel 59' ,'Pixel 60' ,'Pixel 61' ,'Pixel 62' ,'Pixel 63' ,'Thermistor' ,'Timestamp']
###############################################################################
#All Files to Convert

## 0 Person
filename0_1 = 'input_data/_M1_Bahnhof/m1_leer_2018-03-12_17-57-50.csv'
filename0_2 = 'input_data/_M4_Aemattli/m4_leer_2018-03-21_18-44-41.csv'
filename0_3 = 'input_data/_M5_Ennet/m5_leer_2018-03-21_19-15-37.csv'
filename0_4 = 'input_data/0p/v1/0p_output_2018-05-16_17-25-52.csv'
filename0_5 = 'input_data/0p/v1/0p_output_2018-05-16_17-38-14.csv'
filename0_6 = 'input_data/0p/v1/0p_output_2018-05-16_17-43-38.csv'
train_person0_files = [filename0_1, 
                       filename0_2, 
                       filename0_3,
                       filename0_4,
                       filename0_5,
                       filename0_6]

## 1 Person
filename1_1 ='input_data/1p/v3/a_output_2018-05-09_18-19-07.csv'
filename1_2 ='input_data/1p/v3/b_output_2018-05-09_18-20-33.csv'
filename1_3 ='input_data/1p/v3/d_output_2018-05-09_18-21-54.csv'
filename1_4 ='input_data/1p/v3/e_output_2018-05-09_18-23-14.csv'
filename1_5 ='input_data/1p/v3/g_output_2018-05-09_18-24-29.csv'
filename1_6 ='input_data/1p/v3/i_output_2018-05-09_18-26-15.csv'
filename1_7 ='input_data/1p/v3/j_output_2018-05-09_18-27-36.csv'
filename1_8 ='input_data/1p/v4/a_output_2018-05-09_18-41-24.csv'
filename1_9 ='input_data/1p/v4/b_output_2018-05-09_18-42-47.csv'
filename1_10 ='input_data/1p/v4/d_output_2018-05-09_18-44-03.csv'
filename1_11 ='input_data/1p/v4/e_output_2018-05-09_18-45-19.csv'
filename1_12 ='input_data/1p/v4/g_output_2018-05-09_18-46-35.csv'
filename1_13 ='input_data/1p/v4/i_output_2018-05-09_18-47-57.csv'
filename1_14 ='input_data/1p/v4/j_output_2018-05-09_18-49-25.csv'
train_person1_files = [filename1_1, 
                       filename1_2, 
                       filename1_3, 
                       filename1_4, 
                       filename1_5, 
                       filename1_6, 
                       filename1_7, 
                       filename1_8, 
                       filename1_9, 
                       filename1_10,
                       filename1_11, 
                       filename1_12, 
                       filename1_13, 
                       filename1_14]

## 2 Persons
filename2_1 ='input_data/2p/v3/ab_output_2018-05-09_18-51-30.csv'
filename2_2 ='input_data/2p/v3/ac_output_2018-05-09_18-53-00.csv'
filename2_3 ='input_data/2p/v3/ai_output_2018-05-09_18-54-15.csv'
filename2_4 ='input_data/2p/v3/de_output_2018-05-09_18-55-31.csv'
filename2_5 ='input_data/2p/v3/df_output_2018-05-09_18-56-45.csv'
filename2_6 ='input_data/2p/v3/gc_output_2018-05-09_18-58-02.csv'
filename2_7 ='input_data/2p/v3/je_output_2018-05-09_18-59-25.csv'
filename2_8 ='input_data/2p/v3/ji_output_2018-05-09_19-00-42.csv'
filename2_9 ='input_data/2p/v4/ab_output_2018-05-09_18-30-05.csv'
filename2_10 ='input_data/2p/v4/ac_output_2018-05-09_18-31-29.csv'
filename2_11 ='input_data/2p/v4/ac_output_2018-05-09_18-31-29.csv'
filename2_12 ='input_data/2p/v4/de_output_2018-05-09_18-34-11.csv'
filename2_13 ='input_data/2p/v4/df_output_2018-05-09_18-35-39.csv'
filename2_14 ='input_data/2p/v4/gc_output_2018-05-09_18-36-54.csv'
filename2_15='input_data/2p/v4/je_output_2018-05-09_18-38-12.csv'
filename2_16='input_data/2p/v4/ji_output_2018-05-09_18-39-40.csv'
# testfile
filename2_17 ='input_data/_M2_Feld/m2_two_person_175_165_stehend_2018-03-12_18-49-54.csv'
train_person2_files = [filename2_1,
                       filename2_2,
                       filename2_3,
                       filename2_4,
                       filename2_5,
                       filename2_6,
                       filename2_7,
                       filename2_8,
                       filename2_9,
                       filename2_10, 
                       filename2_11, 
                       filename2_12, 
                       filename2_13, 
                       filename2_14, 
                       filename2_15,
                       filename2_16,
                       filename2_17]

## 3 Persons
filename3_1 ='input_data/3p/v2/a_b_c_output_2018-05-09_18-05-08.csv'
filename3_2 ='input_data/3p/v2/a_e_i_output_2018-05-09_18-08-12.csv'
filename3_3 ='input_data/3p/v2/d_e_f_output_2018-05-09_18-06-46.csv'
filename3_4 ='input_data/3p/v2/g_e_i_output_2018-05-09_18-09-48.csv'
filename3_5 ='input_data/3p/v2/g_k_i_output_2018-05-09_18-13-10.csv'
filename3_6 ='input_data/3p/v2/h_d_c_output_2018-05-09_18-16-14.csv'
filename3_7 ='input_data/3p/v2/i_e_a_output_2018-05-09_18-14-56.csv'
filename3_8 ='input_data/3p/v2/j_h_l_output_2018-05-09_18-11-45.csv'
train_person3_files = [filename3_1, 
                       filename3_2, 
                       filename3_3, 
                       filename3_4, 
                       filename3_5, 
                       filename3_6, 
                       filename3_7, 
                       filename3_8]

## 4 Personen
filename4_1 ='input_data/4p/v2/p4_acdf_output_2018-05-09_17-55-36.csv'
filename4_2 ='input_data/4p/v2/p4_acjl_output_2018-05-09_17-53-29.csv'
filename4_3 ='input_data/4p/v2/p4_adgj_output_2018-05-09_17-58-42.csv'
filename4_4 ='input_data/4p/v2/p4_dfac_output_2018-05-09_18-03-01.csv'
filename4_5 ='input_data/4p/v2/p4_dfgi_output_2018-05-09_17-57-09.csv'
filename4_6 ='input_data/4p/v2/p4_gjcf_output_2018-05-09_18-00-24.csv'

train_person4_files = [filename4_1,
                       filename4_2,
                       filename4_3, 
                       filename4_4,
                       filename4_5,
                       filename4_6]

###############################################################################

if swap: 
    swapfkt(train_person0_files)
    swapfkt(train_person1_files)
    swapfkt(train_person2_files)
    swapfkt(train_person3_files)
    swapfkt(train_person4_files)

if rotate90:
    rotate90fkt(train_person0_files)  
    rotate90fkt(train_person1_files)
    rotate90fkt(train_person2_files)
    rotate90fkt(train_person3_files)
    rotate90fkt(train_person4_files)
    
    
if rotate180:
    rotate180fkt(train_person0_files)  
    rotate180fkt(train_person1_files)
    rotate180fkt(train_person2_files)
    rotate180fkt(train_person3_files)
    rotate180fkt(train_person4_files)

    
if rotate270:
    rotate270fkt(train_person0_files)  
    rotate270fkt(train_person1_files)
    rotate270fkt(train_person2_files)
    rotate270fkt(train_person3_files)
    rotate270fkt(train_person4_files)

    
    