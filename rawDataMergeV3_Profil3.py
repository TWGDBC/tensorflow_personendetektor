# -*- coding: utf-8 -*-
# pyhton 2.7 ability
"""
Created on Mon Apr 23 00:56:16 2018

Skript to merge all defined Data to TRAIN, TEST FILES & LABELS
creates Dataset in /prepared_data/...  (4 Files with Timestamp)
Profil 3

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
        
        data = pd.read_csv(value, names=CSV_COLUMN_NAMES, header=0,)
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
filename0_10 = 'input_data/0p/v1/0p_output_2018-05-16_17-43-38_rotateswap.csv'

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
filename1_1 ='input_data/1p/v5/a_output_2018-05-16_17-57-58.csv'
filename1_2 ='input_data/1p/v5/b_output_2018-05-16_18-00-21.csv'
filename1_3 ='input_data/1p/v5/d_output_2018-05-16_18-02-02.csv'
filename1_4 ='input_data/1p/v5/e_output_2018-05-16_18-03-50.csv'
filename1_5 ='input_data/1p/v5/g_output_2018-05-16_18-05-13.csv'
filename1_6 ='input_data/1p/v5/i_output_2018-05-16_18-08-49.csv'
filename1_7 ='input_data/1p/v5/j_output_2018-05-16_18-06-41.csv'
filename1_8 ='input_data/1p/v6/a_output_2018-05-16_18-52-28.csv'
filename1_9 ='input_data/1p/v6/b_output_2018-05-16_18-55-12.csv'
filename1_10 ='input_data/1p/v6/d_output_2018-05-16_18-56-37.csv'
filename1_11 ='input_data/1p/v6/e_output_2018-05-16_18-57-53.csv'
filename1_12 ='input_data/1p/v6/g_output_2018-05-16_18-59-13.csv'
filename1_13 ='input_data/1p/v6/i_output_2018-05-16_19-00-28.csv'
filename1_14 ='input_data/1p/v6/j_output_2018-05-16_19-01-43.csv'
## rotated 90

##_rotate180

##rotate 270

## swap


## Alle Files in Liste anfügen Listenmax 
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
filename2_1 ='input_data/2p/v5/ab_output_2018-05-16_18-10-25.csv'
filename2_2 ='input_data/2p/v5/ac_output_2018-05-16_18-12-01.csv'
filename2_3 ='input_data/2p/v5/ai_output_2018-05-16_18-13-27.csv'
filename2_4 ='input_data/2p/v5/de_output_2018-05-16_18-14-49.csv'
filename2_5 ='input_data/2p/v5/df_output_2018-05-16_18-16-21.csv'
filename2_6 ='input_data/2p/v5/gc_output_2018-05-16_18-17-45.csv'
filename2_7 ='input_data/2p/v5/je_output_2018-05-16_18-19-15.csv'
filename2_8 ='input_data/2p/v5/ji_output_2018-05-16_18-20-41.csv'
filename2_9 ='input_data/2p/v6/ab_output_2018-05-16_18-22-57.csv'
filename2_10 ='input_data/2p/v6/ac_output_2018-05-16_18-24-27.csv'
filename2_11 ='input_data/2p/v6/ai_output_2018-05-16_18-25-49.csv'
filename2_12 ='input_data/2p/v6/de_output_2018-05-16_18-27-14.csv'
filename2_13 ='input_data/2p/v6/df_output_2018-05-16_18-28-36.csv'
filename2_14 ='input_data/2p/v6/gc_output_2018-05-16_18-30-04.csv'
filename2_15='input_data/2p/v6/je_output_2018-05-16_18-31-32.csv'
filename2_16='input_data/2p/v6/ji_output_2018-05-16_18-32-57.csv'
## rotate 90

## rotate 180

## rotate 270

## swap


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
filename3_1 ='input_data/3p/v3/abc_output_2018-05-16_19-04-34.csv'
filename3_2 ='input_data/3p/v3/aei_output_2018-05-16_19-14-42.csv'
filename3_3 ='input_data/3p/v3/def_output_2018-05-16_19-11-32.csv'
filename3_4 ='input_data/3p/v3/dhl_output_2018-05-16_19-18-00.csv'
filename3_5 ='input_data/3p/v3/gei_output_2018-05-16_19-16-05.csv'
filename3_6 ='input_data/3p/v3/gki_output_2018-05-16_19-19-22.csv'
filename3_7 ='input_data/3p/v3/hdc_output_2018-05-16_19-22-34.csv'
filename3_8 ='input_data/3p/v3/iea_output_2018-05-16_19-21-08.csv'
## rotate 90

## rotate 180

## rotate 270

## swap


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
filename4_1 ='input_data/4p/v3/p4_acdf_output_2018-05-16_19-26-26.csv'
filename4_2 ='input_data/4p/v3/p4_acjl_output_2018-05-16_19-24-42.csv'
filename4_3 ='input_data/4p/v3/p4_adgj_output_2018-05-16_19-29-38.csv'
filename4_4 ='input_data/4p/v3/p4_dfac_output_2018-05-16_19-32-52.csv'
filename4_5 ='input_data/4p/v3/p4_dfgi_output_2018-05-16_19-27-54.csv'
filename4_6 ='input_data/4p/v3/p4_gjcf_output_2018-05-16_19-31-12.csv'
## rotate 90

## rotate 180

## rotate 270

## swap


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
filenameT0_1 = 'input_data/0p_rauschen/rauschen_20_min_2018-03-13_14-09-29.csv'
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
