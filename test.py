# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 23:28:54 2018

@author: User
"""
import csv

def readcsv():
  ifile = open('prepared_data/testImages_2018-04-28_13-49-44.csv', "rU")
  reader = csv.reader(ifile, delimiter=",")

  rownum = 0	
  a = []

  for row in reader:
     a.append (row)
     rownum += 1
    
  ifile.close()
  return a


a = readcsv()