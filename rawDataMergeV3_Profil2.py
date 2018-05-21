# -*- coding: utf-8 -*-
# pyhton 2.7 ability
"""
Created on Mon Apr 23 00:56:16 2018

Skript to merge all defined Data to TRAIN, TEST FILES & LABELS
creates Dataset in /prepared_data/...  (4 Files with Timestamp)
Profil 2

@author: Daniel Zimmermann 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import datetime
#loading the data sets from the csv files
print('--------load train & validation & test Files ------')

trainImages = pd.DataFrame()
trainLabels = pd.DataFrame()
testImages = pd.DataFrame()
testLabels = pd.DataFrame()
trainNbr = pd.DataFrame()
tstNbr = pd.DataFrame()

###############################################################################
#Helpfer functions  

def mergeFilesToTrainSet(filelist,labelNumber):
    
    for value in filelist:  
        global trainImages
        global trainLabels
        global trainNbr
        
        data = pd.read_csv(value, names=CSV_COLUMN_NAMES, header=0)
        ## delete Timestamp and Thermistor Data
        data.pop("Timestamp")
        #data=data.rename(columns = {'Timestamp':'Nmbr'})
        data.pop('Thermistor') 
        newTrainLabels = pd.DataFrame(data = np.ones((len(data),1), dtype = np.int8)*labelNumber, columns=['labels'])
        trainImages = pd.concat([trainImages,data],axis = 0)
        #trainImages['Nmbr']=trainImages['Nmbr'].fill(trainImages.index.to_series())
        trainLabels = pd.concat([trainLabels,newTrainLabels],axis = 0)
        
def mergeFilesToTestSet(filelist,labelNumber):
    
    for value in filelist:  
        global testImages
        global testLabels
        global testNbr
        data = pd.read_csv(value, names=CSV_COLUMN_NAMES, header=0,)
        data.pop("Timestamp")
        data.pop('Thermistor')
        newTestLabels = pd.DataFrame(data = np.ones((len(data),1),dtype = np.int8)*labelNumber, columns=['labels'])
        testImages = pd.concat([testImages,data],axis = 0)
        testLabels = pd.concat([testLabels,newTestLabels],axis = 0)   
         
###############################################################################
# read training data from CSV file 
# Column_names
CSV_COLUMN_NAMES = ['Pixel 0' ,'Pixel 1' ,'Pixel 2' ,'Pixel 3' ,'Pixel 4' ,'Pixel 5' ,'Pixel 6' ,'Pixel 7' ,'Pixel 8' ,
'Pixel 9' ,'Pixel 10' ,'Pixel 11' ,'Pixel 12' ,'Pixel 13' ,'Pixel 14' ,'Pixel 15' ,'Pixel 16' ,
'Pixel 17' ,'Pixel 18' ,'Pixel 19' ,'Pixel 20' ,'Pixel 21' ,'Pixel 22' ,'Pixel 23' ,'Pixel 24' ,
'Pixel 25' ,'Pixel 26' ,'Pixel 27' ,'Pixel 28' ,'Pixel 29' ,'Pixel 30' ,'Pixel 31' ,'Pixel 32' ,
'Pixel 33' ,'Pixel 34' ,'Pixel 35' ,'Pixel 36' ,'Pixel 37' ,'Pixel 38' ,'Pixel 39' ,'Pixel 40' ,
'Pixel 41' ,'Pixel 42' ,'Pixel 43' ,'Pixel 44' ,'Pixel 45' ,'Pixel 46' ,'Pixel 47' ,'Pixel 48' ,
'Pixel 49' ,'Pixel 50' ,'Pixel 51' ,'Pixel 52' ,'Pixel 53' ,'Pixel 54' ,'Pixel 55' ,'Pixel 56' , 
'Pixel 57' ,'Pixel 58' ,'Pixel 59' ,'Pixel 60' ,'Pixel 61' ,'Pixel 62' ,'Pixel 63' ,'Thermistor' ,'Timestamp']

## 0 Person
filename0_1 = 'input_data/_M1_Bahnhof/m1_leer_2018-03-12_17-57-50.csv'
filename0_2 = 'input_data/_M4_Aemattli/m4_leer_2018-03-21_18-44-41.csv'
filename0_3 = 'input_data/_M5_Ennet/m5_leer_2018-03-21_19-15-37.csv'
filename0_4 = 'input_data/0p/v1/0p_output_2018-05-16_17-25-52.csv'
filename0_5 = 'input_data/0p/v1/0p_output_2018-05-16_17-38-14.csv'
filename0_6 = 'input_data/0p/v1/0p_output_2018-05-16_17-43-38.csv'
filename0_7 = 'input_data/0p/v1/0p_output_2018-05-16_17-43-38_rotate90.csv'
filename0_8 = 'input_data/0p/v1/0p_output_2018-05-16_17-43-38_rotate180.csv'
filename0_9 = 'input_data/0p/v1/0p_output_2018-05-16_17-43-38_rotate270.csv'
filename0_10 = 'input_data/0p/v1/0p_output_2018-05-16_17-43-38_swap.csv'
train_person0_files = [filename0_1, 
                       filename0_2, 
                       filename0_3,
                       filename0_4,
                       filename0_5,
                       filename0_6,
                       filename0_7,
                       filename0_8,
                       filename0_9,
                       filename0_10]

## 1 Person
filename1_1  ='input_data/1p/v3/a_output_2018-05-09_18-19-07.csv'
filename1_2  ='input_data/1p/v3/b_output_2018-05-09_18-20-33.csv'
filename1_3  ='input_data/1p/v3/d_output_2018-05-09_18-21-54.csv'
filename1_4  ='input_data/1p/v3/e_output_2018-05-09_18-23-14.csv'
filename1_5  ='input_data/1p/v3/g_output_2018-05-09_18-24-29.csv'
filename1_6  ='input_data/1p/v3/i_output_2018-05-09_18-26-15.csv'
filename1_7  ='input_data/1p/v3/j_output_2018-05-09_18-27-36.csv'
filename1_8  ='input_data/1p/v4/a_output_2018-05-09_18-41-24.csv'
filename1_9  ='input_data/1p/v4/b_output_2018-05-09_18-42-47.csv'
filename1_10 ='input_data/1p/v4/d_output_2018-05-09_18-44-03.csv'
filename1_11 ='input_data/1p/v4/e_output_2018-05-09_18-45-19.csv'
filename1_12 ='input_data/1p/v4/g_output_2018-05-09_18-46-35.csv'
filename1_13 ='input_data/1p/v4/i_output_2018-05-09_18-47-57.csv'
filename1_14 ='input_data/1p/v4/j_output_2018-05-09_18-49-25.csv'
## rotated 90
filename1_15 ='input_data/1p/v3/a_output_2018-05-09_18-19-07_rotate90.csv'
filename1_16 ='input_data/1p/v3/b_output_2018-05-09_18-20-33_rotate90.csv'
filename1_17 ='input_data/1p/v3/d_output_2018-05-09_18-21-54_rotate90.csv'
filename1_18 ='input_data/1p/v3/e_output_2018-05-09_18-23-14_rotate90.csv'
filename1_19 ='input_data/1p/v3/g_output_2018-05-09_18-24-29_rotate90.csv'
filename1_20 ='input_data/1p/v3/i_output_2018-05-09_18-26-15_rotate90.csv'
filename1_21 ='input_data/1p/v3/j_output_2018-05-09_18-27-36_rotate90.csv'
filename1_22 ='input_data/1p/v4/a_output_2018-05-09_18-41-24_rotate90.csv'
filename1_23 ='input_data/1p/v4/b_output_2018-05-09_18-42-47_rotate90.csv'
filename1_24 ='input_data/1p/v4/d_output_2018-05-09_18-44-03_rotate90.csv'
filename1_25 ='input_data/1p/v4/e_output_2018-05-09_18-45-19_rotate90.csv'
filename1_26 ='input_data/1p/v4/g_output_2018-05-09_18-46-35_rotate90.csv'
filename1_27 ='input_data/1p/v4/i_output_2018-05-09_18-47-57_rotate90.csv'
filename1_28 ='input_data/1p/v4/j_output_2018-05-09_18-49-25_rotate90.csv'
## rotated 180
filename1_29 ='input_data/1p/v3/a_output_2018-05-09_18-19-07_rotate180.csv'
filename1_30 ='input_data/1p/v3/b_output_2018-05-09_18-20-33_rotate180.csv'
filename1_31 ='input_data/1p/v3/d_output_2018-05-09_18-21-54_rotate180.csv'
filename1_32 ='input_data/1p/v3/e_output_2018-05-09_18-23-14_rotate180.csv'
filename1_33 ='input_data/1p/v3/g_output_2018-05-09_18-24-29_rotate180.csv'
filename1_34 ='input_data/1p/v3/i_output_2018-05-09_18-26-15_rotate180.csv'
filename1_35 ='input_data/1p/v3/j_output_2018-05-09_18-27-36_rotate180.csv'
filename1_36 ='input_data/1p/v4/a_output_2018-05-09_18-41-24_rotate180.csv'
filename1_37 ='input_data/1p/v4/b_output_2018-05-09_18-42-47_rotate180.csv'
filename1_38 ='input_data/1p/v4/d_output_2018-05-09_18-44-03_rotate180.csv'
filename1_39 ='input_data/1p/v4/e_output_2018-05-09_18-45-19_rotate180.csv'
filename1_40 ='input_data/1p/v4/g_output_2018-05-09_18-46-35_rotate180.csv'
filename1_41 ='input_data/1p/v4/i_output_2018-05-09_18-47-57_rotate180.csv'
filename1_42 ='input_data/1p/v4/j_output_2018-05-09_18-49-25_rotate180.csv'
## rotated 270
filename1_43 ='input_data/1p/v3/a_output_2018-05-09_18-19-07_rotate270.csv'
filename1_44 ='input_data/1p/v3/b_output_2018-05-09_18-20-33_rotate270.csv'
filename1_45 ='input_data/1p/v3/d_output_2018-05-09_18-21-54_rotate270.csv'
filename1_46 ='input_data/1p/v3/e_output_2018-05-09_18-23-14_rotate270.csv'
filename1_47 ='input_data/1p/v3/g_output_2018-05-09_18-24-29_rotate270.csv'
filename1_48 ='input_data/1p/v3/i_output_2018-05-09_18-26-15_rotate270.csv'
filename1_49 ='input_data/1p/v3/j_output_2018-05-09_18-27-36_rotate270.csv'
filename1_50 ='input_data/1p/v4/a_output_2018-05-09_18-41-24_rotate270.csv'
filename1_51 ='input_data/1p/v4/b_output_2018-05-09_18-42-47_rotate270.csv'
filename1_52 ='input_data/1p/v4/d_output_2018-05-09_18-44-03_rotate270.csv'
filename1_53 ='input_data/1p/v4/e_output_2018-05-09_18-45-19_rotate270.csv'
filename1_54 ='input_data/1p/v4/g_output_2018-05-09_18-46-35_rotate270.csv'
filename1_55 ='input_data/1p/v4/i_output_2018-05-09_18-47-57_rotate270.csv'
filename1_56 ='input_data/1p/v4/j_output_2018-05-09_18-49-25_rotate270.csv'
## rotated swap
filename1_57 ='input_data/1p/v3/a_output_2018-05-09_18-19-07_swap.csv'
filename1_58 ='input_data/1p/v3/b_output_2018-05-09_18-20-33_swap.csv'
filename1_59 ='input_data/1p/v3/d_output_2018-05-09_18-21-54_swap.csv'
filename1_60 ='input_data/1p/v3/e_output_2018-05-09_18-23-14_swap.csv'
filename1_61 ='input_data/1p/v3/g_output_2018-05-09_18-24-29_swap.csv'
filename1_62 ='input_data/1p/v3/i_output_2018-05-09_18-26-15_swap.csv'
filename1_63 ='input_data/1p/v3/j_output_2018-05-09_18-27-36_swap.csv'
filename1_64 ='input_data/1p/v4/a_output_2018-05-09_18-41-24_swap.csv'
filename1_65 ='input_data/1p/v4/b_output_2018-05-09_18-42-47_swap.csv'
filename1_66 ='input_data/1p/v4/d_output_2018-05-09_18-44-03_swap.csv'
filename1_67 ='input_data/1p/v4/e_output_2018-05-09_18-45-19_swap.csv'
filename1_68 ='input_data/1p/v4/g_output_2018-05-09_18-46-35_swap.csv'
filename1_69 ='input_data/1p/v4/i_output_2018-05-09_18-47-57_swap.csv'
filename1_70 ='input_data/1p/v4/j_output_2018-05-09_18-49-25_swap.csv'

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
                       filename1_14, 
                       filename1_15,
                       filename1_16,
                       filename1_17,
                       filename1_18,
                       filename1_19,
                       filename1_20,
                       filename1_21,
                       filename1_22, 
                       filename1_23,
                       filename1_24,
                       filename1_25, 
                       filename1_26, 
                       filename1_27,
                       filename1_28, 
                       filename1_29,
                       filename1_30,
                       filename1_31,
                       filename1_32, 
                       filename1_33,
                       filename1_34,
                       filename1_35, 
                       filename1_36, 
                       filename1_37,
                       filename1_38, 
                       filename1_39,
                       filename1_40,
                       filename1_41,
                       filename1_42,
                       filename1_43,
                       filename1_44, 
                       filename1_45,
                       filename1_46,
                       filename1_47,
                       filename1_48,
                       filename1_49,
                       filename1_50,
                       filename1_51,
                       filename1_52,
                       filename1_53,
                       filename1_54, 
                       filename1_55,
                       filename1_56,
                       filename1_57,
                       filename1_58,
                       filename1_59,
                       filename1_60,
                       filename1_61,
                       filename1_62, 
                       filename1_63,
                       filename1_64,
                       filename1_65, 
                       filename1_66, 
                       filename1_67,
                       filename1_68, 
                       filename1_69,
                       filename1_70]

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
# rotate 90
filename2_17 ='input_data/2p/v3/ab_output_2018-05-09_18-51-30_rotate90.csv'
filename2_18 ='input_data/2p/v3/ac_output_2018-05-09_18-53-00_rotate90.csv'
filename2_19 ='input_data/2p/v3/ai_output_2018-05-09_18-54-15_rotate90.csv'
filename2_20 ='input_data/2p/v3/de_output_2018-05-09_18-55-31_rotate90.csv'
filename2_21 ='input_data/2p/v3/df_output_2018-05-09_18-56-45_rotate90.csv'
filename2_22 ='input_data/2p/v3/gc_output_2018-05-09_18-58-02_rotate90.csv'
filename2_23 ='input_data/2p/v3/je_output_2018-05-09_18-59-25_rotate90.csv'
filename2_24 ='input_data/2p/v3/ji_output_2018-05-09_19-00-42_rotate90.csv'
filename2_25 ='input_data/2p/v4/ab_output_2018-05-09_18-30-05_rotate90.csv'
filename2_26 ='input_data/2p/v4/ac_output_2018-05-09_18-31-29_rotate90.csv'
filename2_27 ='input_data/2p/v4/ac_output_2018-05-09_18-31-29_rotate90.csv'
filename2_28 ='input_data/2p/v4/de_output_2018-05-09_18-34-11_rotate90.csv'
filename2_29 ='input_data/2p/v4/df_output_2018-05-09_18-35-39_rotate90.csv'
filename2_30 ='input_data/2p/v4/gc_output_2018-05-09_18-36-54_rotate90.csv'
filename2_31 ='input_data/2p/v4/je_output_2018-05-09_18-38-12_rotate90.csv'
filename2_32 ='input_data/2p/v4/ji_output_2018-05-09_18-39-40_rotate90.csv'
# rotate 180
filename2_33 ='input_data/2p/v3/ab_output_2018-05-09_18-51-30_rotate180.csv'
filename2_34 ='input_data/2p/v3/ac_output_2018-05-09_18-53-00_rotate180.csv'
filename2_35 ='input_data/2p/v3/ai_output_2018-05-09_18-54-15_rotate180.csv'
filename2_36 ='input_data/2p/v3/de_output_2018-05-09_18-55-31_rotate180.csv'
filename2_37 ='input_data/2p/v3/df_output_2018-05-09_18-56-45_rotate180.csv'
filename2_38 ='input_data/2p/v3/gc_output_2018-05-09_18-58-02_rotate180.csv'
filename2_39 ='input_data/2p/v3/je_output_2018-05-09_18-59-25_rotate180.csv'
filename2_40 ='input_data/2p/v3/ji_output_2018-05-09_19-00-42_rotate180.csv'
filename2_41 ='input_data/2p/v4/ab_output_2018-05-09_18-30-05_rotate180.csv'
filename2_42 ='input_data/2p/v4/ac_output_2018-05-09_18-31-29_rotate180.csv'
filename2_43 ='input_data/2p/v4/ac_output_2018-05-09_18-31-29_rotate180.csv'
filename2_44 ='input_data/2p/v4/de_output_2018-05-09_18-34-11_rotate180.csv'
filename2_45 ='input_data/2p/v4/df_output_2018-05-09_18-35-39_rotate180.csv'
filename2_46 ='input_data/2p/v4/gc_output_2018-05-09_18-36-54_rotate180.csv'
filename2_47 ='input_data/2p/v4/je_output_2018-05-09_18-38-12_rotate180.csv'
filename2_48 ='input_data/2p/v4/ji_output_2018-05-09_18-39-40_rotate180.csv'
#rotate 270
filename2_49 ='input_data/2p/v3/ab_output_2018-05-09_18-51-30_rotate270.csv'
filename2_50 ='input_data/2p/v3/ac_output_2018-05-09_18-53-00_rotate270.csv'
filename2_51 ='input_data/2p/v3/ai_output_2018-05-09_18-54-15_rotate270.csv'
filename2_52 ='input_data/2p/v3/de_output_2018-05-09_18-55-31_rotate270.csv'
filename2_53 ='input_data/2p/v3/df_output_2018-05-09_18-56-45_rotate270.csv'
filename2_54 ='input_data/2p/v3/gc_output_2018-05-09_18-58-02_rotate270.csv'
filename2_55 ='input_data/2p/v3/je_output_2018-05-09_18-59-25_rotate270.csv'
filename2_56 ='input_data/2p/v3/ji_output_2018-05-09_19-00-42_rotate270.csv'
filename2_57 ='input_data/2p/v4/ab_output_2018-05-09_18-30-05_rotate270.csv'
filename2_58 ='input_data/2p/v4/ac_output_2018-05-09_18-31-29_rotate270.csv'
filename2_59 ='input_data/2p/v4/ac_output_2018-05-09_18-31-29_rotate270.csv'
filename2_60 ='input_data/2p/v4/de_output_2018-05-09_18-34-11_rotate270.csv'
filename2_61 ='input_data/2p/v4/df_output_2018-05-09_18-35-39_rotate270.csv'
filename2_62 ='input_data/2p/v4/gc_output_2018-05-09_18-36-54_rotate270.csv'
filename2_63 ='input_data/2p/v4/je_output_2018-05-09_18-38-12_rotate270.csv'
filename2_64 ='input_data/2p/v4/ji_output_2018-05-09_18-39-40_rotate270.csv'
# swap
filename2_65 ='input_data/2p/v3/ab_output_2018-05-09_18-51-30_swap.csv'
filename2_66 ='input_data/2p/v3/ac_output_2018-05-09_18-53-00_swap.csv'
filename2_67 ='input_data/2p/v3/ai_output_2018-05-09_18-54-15_swap.csv'
filename2_68 ='input_data/2p/v3/de_output_2018-05-09_18-55-31_swap.csv'
filename2_69 ='input_data/2p/v3/df_output_2018-05-09_18-56-45_swap.csv'
filename2_70 ='input_data/2p/v3/gc_output_2018-05-09_18-58-02_swap.csv'
filename2_71 ='input_data/2p/v3/je_output_2018-05-09_18-59-25_swap.csv'
filename2_72 ='input_data/2p/v3/ji_output_2018-05-09_19-00-42_swap.csv'
filename2_73 ='input_data/2p/v4/ab_output_2018-05-09_18-30-05_swap.csv'
filename2_74 ='input_data/2p/v4/ac_output_2018-05-09_18-31-29_swap.csv'
filename2_75 ='input_data/2p/v4/ac_output_2018-05-09_18-31-29_swap.csv'
filename2_76 ='input_data/2p/v4/de_output_2018-05-09_18-34-11_swap.csv'
filename2_77 ='input_data/2p/v4/df_output_2018-05-09_18-35-39_swap.csv'
filename2_78 ='input_data/2p/v4/gc_output_2018-05-09_18-36-54_swap.csv'
filename2_79 ='input_data/2p/v4/je_output_2018-05-09_18-38-12_swap.csv'
filename2_80 ='input_data/2p/v4/ji_output_2018-05-09_18-39-40_swap.csv'

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
                       filename2_17,
                       filename2_18,
                       filename2_19,
                       filename2_20,
                       filename2_21,
                       filename2_22, 
                       filename2_23,
                       filename2_24,
                       filename2_25, 
                       filename2_26, 
                       filename2_27,
                       filename2_28, 
                       filename2_29,
                       filename2_30,
                       filename2_31,
                       filename2_32, 
                       filename2_33,
                       filename2_34,
                       filename2_35, 
                       filename2_36, 
                       filename2_37,
                       filename2_38, 
                       filename2_39,
                       filename2_40,
                       filename2_41,
                       filename2_42,
                       filename2_43,
                       filename2_44, 
                       filename2_45,
                       filename2_46,
                       filename2_47,
                       filename2_48,
                       filename2_49,
                       filename2_50,
                       filename2_51,
                       filename2_52,
                       filename2_53,
                       filename2_54, 
                       filename2_55,
                       filename2_56,
                       filename2_57,
                       filename2_58,
                       filename2_59,
                       filename2_60,
                       filename2_61,
                       filename2_62, 
                       filename2_63,
                       filename2_64,
                       filename2_65, 
                       filename2_66, 
                       filename2_67,
                       filename2_68, 
                       filename2_69,
                       filename2_70,
                       filename2_71,
                       filename2_72, 
                       filename2_73,
                       filename2_74,
                       filename2_75, 
                       filename2_76, 
                       filename2_77,
                       filename2_78, 
                       filename2_79,
                       filename2_80] 

## 3 Persons
filename3_1 ='input_data/3p/v2/a_b_c_output_2018-05-09_18-05-08.csv'
filename3_2 ='input_data/3p/v2/a_e_i_output_2018-05-09_18-08-12.csv'
filename3_3 ='input_data/3p/v2/d_e_f_output_2018-05-09_18-06-46.csv'
filename3_4 ='input_data/3p/v2/g_e_i_output_2018-05-09_18-09-48.csv'
filename3_5 ='input_data/3p/v2/g_k_i_output_2018-05-09_18-13-10.csv'
filename3_6 ='input_data/3p/v2/h_d_c_output_2018-05-09_18-16-14.csv'
filename3_7 ='input_data/3p/v2/i_e_a_output_2018-05-09_18-14-56.csv'
filename3_8 ='input_data/3p/v2/j_h_l_output_2018-05-09_18-11-45.csv'
# rotate 90
filename3_9  ='input_data/3p/v2/a_b_c_output_2018-05-09_18-05-08_rotate90.csv'
filename3_10 ='input_data/3p/v2/a_e_i_output_2018-05-09_18-08-12_rotate90.csv'
filename3_11 ='input_data/3p/v2/d_e_f_output_2018-05-09_18-06-46_rotate90.csv'
filename3_12 ='input_data/3p/v2/g_e_i_output_2018-05-09_18-09-48_rotate90.csv'
filename3_13 ='input_data/3p/v2/g_k_i_output_2018-05-09_18-13-10_rotate90.csv'
filename3_14 ='input_data/3p/v2/h_d_c_output_2018-05-09_18-16-14_rotate90.csv'
filename3_15 ='input_data/3p/v2/i_e_a_output_2018-05-09_18-14-56_rotate90.csv'
filename3_16 ='input_data/3p/v2/j_h_l_output_2018-05-09_18-11-45_rotate90.csv'
# roate 180
filename3_17  ='input_data/3p/v2/a_b_c_output_2018-05-09_18-05-08_rotate180.csv'
filename3_18 ='input_data/3p/v2/a_e_i_output_2018-05-09_18-08-12_rotate180.csv'
filename3_19 ='input_data/3p/v2/d_e_f_output_2018-05-09_18-06-46_rotate180.csv'
filename3_20 ='input_data/3p/v2/g_e_i_output_2018-05-09_18-09-48_rotate180.csv'
filename3_21 ='input_data/3p/v2/g_k_i_output_2018-05-09_18-13-10_rotate180.csv'
filename3_22 ='input_data/3p/v2/h_d_c_output_2018-05-09_18-16-14_rotate180.csv'
filename3_23 ='input_data/3p/v2/i_e_a_output_2018-05-09_18-14-56_rotate180.csv'
filename3_24 ='input_data/3p/v2/j_h_l_output_2018-05-09_18-11-45_rotate180.csv'
# rotate 270
filename3_25 ='input_data/3p/v2/a_b_c_output_2018-05-09_18-05-08_rotate270.csv'
filename3_26 ='input_data/3p/v2/a_e_i_output_2018-05-09_18-08-12_rotate270.csv'
filename3_27 ='input_data/3p/v2/d_e_f_output_2018-05-09_18-06-46_rotate270.csv'
filename3_28 ='input_data/3p/v2/g_e_i_output_2018-05-09_18-09-48_rotate270.csv'
filename3_29 ='input_data/3p/v2/g_k_i_output_2018-05-09_18-13-10_rotate270.csv'
filename3_30 ='input_data/3p/v2/h_d_c_output_2018-05-09_18-16-14_rotate270.csv'
filename3_31 ='input_data/3p/v2/i_e_a_output_2018-05-09_18-14-56_rotate270.csv'
filename3_32 ='input_data/3p/v2/j_h_l_output_2018-05-09_18-11-45_rotate270.csv'
# swap
filename3_33 ='input_data/3p/v2/a_b_c_output_2018-05-09_18-05-08_swap.csv'
filename3_34 ='input_data/3p/v2/a_e_i_output_2018-05-09_18-08-12_swap.csv'
filename3_35 ='input_data/3p/v2/d_e_f_output_2018-05-09_18-06-46_swap.csv'
filename3_36 ='input_data/3p/v2/g_e_i_output_2018-05-09_18-09-48_swap.csv'
filename3_37 ='input_data/3p/v2/g_k_i_output_2018-05-09_18-13-10_swap.csv'
filename3_38 ='input_data/3p/v2/h_d_c_output_2018-05-09_18-16-14_swap.csv'
filename3_39 ='input_data/3p/v2/i_e_a_output_2018-05-09_18-14-56_swap.csv'
filename3_40 ='input_data/3p/v2/j_h_l_output_2018-05-09_18-11-45_swap.csv'

train_person3_files = [filename3_1,
                       filename3_2,
                       filename3_3,
                       filename3_4,
                       filename3_5,
                       filename3_6,
                       filename3_7,
                       filename3_8,
                       filename3_9,
                       filename3_10,
                       filename3_11,
                       filename3_12,
                       filename3_13,
                       filename3_14, 
                       filename3_15,
                       filename3_16,
                       filename3_17,
                       filename3_18,
                       filename3_19,
                       filename3_20,
                       filename3_21,
                       filename3_22, 
                       filename3_23,
                       filename3_24,
                       filename3_25, 
                       filename3_26, 
                       filename3_27,
                       filename3_28, 
                       filename3_29,
                       filename3_30,
                       filename3_31,
                       filename3_32, 
                       filename3_33,
                       filename3_34,
                       filename3_35, 
                       filename3_36, 
                       filename3_37,
                       filename3_38, 
                       filename3_39,
                       filename3_40]

## 4 Personen
filename4_1 ='input_data/4p/v2/p4_acdf_output_2018-05-09_17-55-36.csv'
filename4_2 ='input_data/4p/v2/p4_acjl_output_2018-05-09_17-53-29.csv'
filename4_3 ='input_data/4p/v2/p4_adgj_output_2018-05-09_17-58-42.csv'
filename4_4 ='input_data/4p/v2/p4_dfac_output_2018-05-09_18-03-01.csv'
filename4_5 ='input_data/4p/v2/p4_dfgi_output_2018-05-09_17-57-09.csv'
filename4_6 ='input_data/4p/v2/p4_gjcf_output_2018-05-09_18-00-24.csv'
# _rotate 90
filename4_7 ='input_data/4p/v2/p4_acdf_output_2018-05-09_17-55-36_rotate90.csv'
filename4_8 ='input_data/4p/v2/p4_acjl_output_2018-05-09_17-53-29_rotate90.csv'
filename4_9 ='input_data/4p/v2/p4_adgj_output_2018-05-09_17-58-42_rotate90.csv'
filename4_10 ='input_data/4p/v2/p4_dfac_output_2018-05-09_18-03-01_rotate90.csv'
filename4_11 ='input_data/4p/v2/p4_dfgi_output_2018-05-09_17-57-09_rotate90.csv'
filename4_12 ='input_data/4p/v2/p4_gjcf_output_2018-05-09_18-00-24_rotate90.csv'
# rotate 180
filename4_13 ='input_data/4p/v2/p4_acdf_output_2018-05-09_17-55-36_rotate180.csv'
filename4_14 ='input_data/4p/v2/p4_acjl_output_2018-05-09_17-53-29_rotate180.csv'
filename4_15 ='input_data/4p/v2/p4_adgj_output_2018-05-09_17-58-42_rotate180.csv'
filename4_16 ='input_data/4p/v2/p4_dfac_output_2018-05-09_18-03-01_rotate180.csv'
filename4_17 ='input_data/4p/v2/p4_dfgi_output_2018-05-09_17-57-09_rotate180.csv'
filename4_18 ='input_data/4p/v2/p4_gjcf_output_2018-05-09_18-00-24_rotate180.csv'
# rotate 270
filename4_19 ='input_data/4p/v2/p4_acdf_output_2018-05-09_17-55-36_rotate270.csv'
filename4_20 ='input_data/4p/v2/p4_acjl_output_2018-05-09_17-53-29_rotate270.csv'
filename4_21 ='input_data/4p/v2/p4_adgj_output_2018-05-09_17-58-42_rotate270.csv'
filename4_22 ='input_data/4p/v2/p4_dfac_output_2018-05-09_18-03-01_rotate270.csv'
filename4_23 ='input_data/4p/v2/p4_dfgi_output_2018-05-09_17-57-09_rotate270.csv'
filename4_24 ='input_data/4p/v2/p4_gjcf_output_2018-05-09_18-00-24_rotate270.csv'
# swap
filename4_25 ='input_data/4p/v2/p4_acdf_output_2018-05-09_17-55-36_swap.csv'
filename4_26 ='input_data/4p/v2/p4_acjl_output_2018-05-09_17-53-29_swap.csv'
filename4_27 ='input_data/4p/v2/p4_adgj_output_2018-05-09_17-58-42_swap.csv'
filename4_28 ='input_data/4p/v2/p4_dfac_output_2018-05-09_18-03-01_swap.csv'
filename4_29 ='input_data/4p/v2/p4_dfgi_output_2018-05-09_17-57-09_swap.csv'
filename4_30 ='input_data/4p/v2/p4_gjcf_output_2018-05-09_18-00-24_swap.csv'

train_person4_files = [filename4_1,
                       filename4_2,
                       filename4_3,
                       filename4_4,
                       filename4_5,
                       filename4_6,
                       filename4_7,
                       filename4_8,
                       filename4_9,
                       filename4_10,
                       filename4_11,
                       filename4_12,
                       filename4_13,
                       filename4_14, 
                       filename4_15,
                       filename4_16,
                       filename4_17,
                       filename4_18,
                       filename4_19,
                       filename4_20,
                       filename4_21,
                       filename4_22, 
                       filename4_23,
                       filename4_24,
                       filename4_25, 
                       filename4_26, 
                       filename4_27,
                       filename4_28, 
                       filename4_29,
                       filename4_30]

################################################################
# test set
# choice Features


## 0 Person
filenameT0_1 = 'input_data/0p/rauschen_20_min_2018-03-13_14-09-29.csv'
test_person0_files = [filenameT0_1]


## 1 Person
filenameT1_1 = 'input_data/_M1_Bahnhof/m1_person_165_stehend_2018-03-12_17-59-25.csv'
filenameT1_2 = 'input_data/_M1_Bahnhof/m1_person_175_stehend_2018-03-12_18-00-41.csv'
filenameT1_3 = 'input_data/_M3_Feld_5/m3_person_165_stehend_2018-03-21_18-14-26.csv'
filenameT1_4 = 'input_data/_M5_Ennet/m5_person_165_stehend_2018-03-21_19-12-18.csv'

test_person1_files = [filenameT1_1,
                      filenameT1_2,
                      filenameT1_3, 
                      filenameT1_4]

## 2 Persons
filenameT2_1= 'input_data/_M2_Feld/m2_two_person_175_165_stehend_2018-03-12_18-49-54.csv'
test_person2_files = [filenameT2_1]

## 3 Persons
test_person3_files = []



## 4 Persons
test_person4_files = []

###############################################################################
# train set
# merge all prepared train data
mergeFilesToTrainSet(train_person0_files,0)
mergeFilesToTrainSet(train_person1_files,1)
mergeFilesToTrainSet(train_person2_files,2)
mergeFilesToTrainSet(train_person3_files,3)
mergeFilesToTrainSet(train_person4_files,4)

## adds image NMBR
#trainImages.insert(64, 'Nmbr', range(0, 0 + len(trainImages)))
#trainLabels.insert(1, 'Nmbr', range(0, 0 + len(trainLabels)))

###############################################################################
# test set
# merge all prepared test data

mergeFilesToTestSet(test_person0_files,0)
mergeFilesToTestSet(test_person1_files,1)
mergeFilesToTestSet(test_person2_files,2)
#mergeFilesToTestSet(test_person3_files,3)
#mergeFilesToTestSet(test_person4_files,4)

## adds Label NMBR of Persons
#testImages.insert(64, 'Nmbr', range(0, 0 + len(testImages)))
#testLabels.insert(1, 'Nmbr', range(0, 0 + len(testLabels)))

###############################################################################
# Store to 

testImages.to_csv('prepared_data/testImages_'+
                  str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))+
                  '.csv',
                  sep=',',
                  encoding='utf-8',
                  index=False)
trainImages.to_csv('prepared_data/trainImages_'+
                  str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))+
                   '.csv',
                   sep=',',
                   encoding='utf-8',
                   index=False)
testLabels.to_csv('prepared_data/testLabels_'+
                  str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))+#
                  '.csv',
                  sep=',',
                  encoding='utf-8',
                  index=False)
trainLabels.to_csv('prepared_data/trainLabels_'+
                   str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))+
                   '.csv',
                   sep=',',

                   encoding='utf-8',
                   index=False)

print("------finish Generating --------------------")
