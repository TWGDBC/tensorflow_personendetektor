# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 09:55:13 2018

@author: User
"""
import pandas as pd
import numpy as np

df = pd.DataFrame({"A":["foo", "foo", "foo", "bar"], "B":[0,1,1,1], "C":["A","A","B","A"]})
#df.drop_duplicates(subset=['A', 'C'], keep=False)
print(pd.DataFrame.values)
