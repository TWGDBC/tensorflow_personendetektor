# -*- coding: utf-8 -*-
"""
Created on Tue April  18 09:33:35 2018

Skript to Rotate and Swap defined Data, to create more Datasets
Profi 1
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
def rotate270fkt(filelist):  
   for value in filelist: 
          data = pd.read_csv(value, names=CSV_COLUMN_NAMES, header=1)
          Data180 = pd.DataFrame()
          Data180 = data[['Pixel 56' ,'Pixel 48' ,'Pixel 40' ,'Pixel 32' ,'Pixel 24' ,'Pixel 16' ,'Pixel 8' ,'Pixel 0',
                         'Pixel 57' ,'Pixel 49' ,'Pixel 41' ,'Pixel 33' ,'Pixel 25' ,'Pixel 17' ,'Pixel 9' ,'Pixel 1',
                         'Pixel 58' ,'Pixel 50' ,'Pixel 42' ,'Pixel 34' ,'Pixel 26' ,'Pixel 18' ,'Pixel 10' ,'Pixel 2',
                         'Pixel 59' ,'Pixel 51' ,'Pixel 43' ,'Pixel 35' ,'Pixel 27' ,'Pixel 19' ,'Pixel 11' ,'Pixel 3',
                         'Pixel 60' ,'Pixel 52' ,'Pixel 44' ,'Pixel 36' ,'Pixel 28' ,'Pixel 20' ,'Pixel 12' ,'Pixel 4',
                         'Pixel 61' ,'Pixel 53' ,'Pixel 45' ,'Pixel 37' ,'Pixel 29' ,'Pixel 21' ,'Pixel 13' ,'Pixel 5',
                         'Pixel 62' ,'Pixel 54' ,'Pixel 46' ,'Pixel 38' ,'Pixel 30' ,'Pixel 22' ,'Pixel 14' ,'Pixel 6',
                         'Pixel 63' ,'Pixel 55' ,'Pixel 47' ,'Pixel 39' ,'Pixel 31' ,'Pixel 23' ,'Pixel 15' ,'Pixel 7',
                         'Thermistor' ,'Timestamp']]
          Data180.to_csv(value[:-4]+'_rotate270.csv',
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
train_person0_files = [filename0_1, 
                       filename0_2, 
                       filename0_3]

## 1 Person
filename1_1 ='input_data/1p/v2/g_a_output_2018-03-28_19-24-16.csv'
filename1_2 ='input_data/1p/v2/g_b_output_2018-03-28_19-32-40.csv'
filename1_3 ='input_data/1p/v2/g_d_output_2018-03-28_19-31-22.csv'
filename1_4 ='input_data/1p/v2/g_e_output_2018-03-28_19-25-38.csv'
filename1_5 ='input_data/1p/v2/g_g_output_2018-03-28_19-27-11.csv'
filename1_6 ='input_data/1p/v2/g_i_output_2018-03-28_19-28-44.csv'
filename1_7 ='input_data/1p/v2/g_j_output_2018-03-28_19-30-02.csv'
filename1_8 ='input_data/1p/v1/m_a_output_2018-03-28_18-13-28.csv'
filename1_9 ='input_data/1p/v1/m_b_output_2018-03-28_18-19-22.csv'
filename1_10 ='input_data/1p/v1/m_d_output_2018-03-28_18-16-57.csv'
filename1_11 ='input_data/1p/v1/m_e_output_2018-03-28_18-29-26.csv'
filename1_12 ='input_data/1p/v1/m_g_output_2018-03-28_18-24-14.csv'
filename1_13 ='input_data/1p/v1/m_i_output_2018-03-28_18-26-04.csv'
filename1_14 ='input_data/1p/v1/m_j_output_2018-03-28_18-27-48.csv'
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
filename2_1 ='input_data/2p/v1/k_a_g_b_output_2018-03-28_18-35-06.csv'
filename2_2 ='input_data/2p/v1/k_a_g_c_output_2018-03-28_18-32-43.csv'
filename2_3 ='input_data/2p/v1/k_a_g_i_output_2018-03-28_18-42-08.csv'
filename2_4 ='input_data/2p/v1/k_d_g_e_output_2018-03-28_18-38-29.csv'
filename2_5 ='input_data/2p/v1/k_d_g_f_output_2018-03-28_18-36-53.csv'
filename2_6 ='input_data/2p/v1/k_g_g_c_output_2018-03-28_18-40-21.csv'
filename2_7 ='input_data/2p/v1/k_j_g_e_output_2018-03-28_18-43-34.csv'
filename2_8 ='input_data/2p/v1/k_j_g_i_output_2018-03-28_18-45-04.csv'
filename2_9 ='input_data/2p/v2/a_b_output_2018-03-28_19-36-43.csv'
filename2_10 ='input_data/2p/v2/a_c_output_2018-03-28_19-35-19.csv'
filename2_11 ='input_data/2p/v2/a_i_output_2018-03-28_19-42-19.csv'
filename2_12 ='input_data/2p/v2/d_e_output_2018-03-28_19-39-32.csv'
filename2_13 ='input_data/2p/v2/d_f_output_2018-03-28_19-38-07.csv'
filename2_14 ='input_data/2p/v2/g_c_output_2018-03-28_19-40-57.csv'
filename2_15='input_data/2p/v2/j_e_output_2018-03-28_19-43-38.csv'
filename2_16='input_data/2p/v2/j_i_output_2018-03-28_19-45-03.csv'
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
                       filename2_16]

## 3 Persons
filename3_1 ='input_data/3p/v1/k_a_m_b_k_c_output_2018-03-28_18-47-19.csv'
filename3_2 ='input_data/3p/v1/k_a_m_e_k_i_output_2018-03-28_18-52-43.csv'
filename3_3 ='input_data/3p/v1/k_d_m_e_k_f_output_2018-03-28_18-49-21.csv'
filename3_4 ='input_data/3p/v1/k_g_m_e_k_i_output_2018-03-28_18-54-39.csv'
filename3_5 ='input_data/3p/v1/k_a_g_e_k_i_output_2018-03-28_19-18-52.csv'
filename3_6 ='input_data/3p/v1/k_j_m_h_k_l_output_2018-03-28_18-56-51.csv'
filename3_7 ='input_data/3p/v1/k_g_m_k_k_i_output_2018-03-28_18-58-23.csv'
filename3_8 ='input_data/3p/v1/k_h_g_d_k_loutput_2018-03-28_19-20-56.csv'
train_person3_files = [filename3_1, 
                       filename3_2, 
                       filename3_3, 
                       filename3_4, 
                       filename3_5, 
                       filename3_6, 
                       filename3_7, 
                       filename3_8]

## 4 Personen
filename4_1 ='input_data/4p/v1/p4_acdf_output_2018-03-28_19-02-54.csv'
filename4_2 ='input_data/4p/v1/p4_gjcf_output_2018-03-28_19-09-47.csv'
filename4_3 ='input_data/4p/v1/p4_acjl_output_2018-03-28_19-00-55.csv'
filename4_4 ='input_data/4p/v1/p4_acdf_output_2018-03-28_19-04-29.csv'
filename4_5 ='input_data/4p/v1/p4_dfgi_output_2018-03-28_19-06-25.csv'
filename4_6 ='input_data/4p/v1/p4_adgj_output_2018-03-28_19-08-05.csv'
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

    
    