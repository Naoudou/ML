# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:21:29 2023

@author: pcsf28
"""

import pandas as pd

data1=pd.read_csv("Wholesale customers data.csv")
#data1.describe()
data_100=data1.loc['Channel':'Delicassen']
data_100
data_100.describe()
c=data_100.loc["Channel":"Delicassen"]
c
data1.iloc[0:3]

