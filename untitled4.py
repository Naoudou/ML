# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:59:13 2023

@author: pcsf28
"""
import pandas as pd



surveys_df = pd.read_csv("data2.csv")
surveys_df 
gr=surveys_df["Age"].plot()

surveys_df.iloc[:,['Name','Salary','Year']]

surveys_df.iloc[0:3,1:4]

#select all column from row af index values 0 and 7
surveys_df.iloc[[0,7],:]

#Select second Row and 4th column
surveys_df.iloc[2:4]

#some more exemple
surveys_df.iloc[0:3]
surveys_df.iloc[:5]
surveys_df.iloc[-1:]

surveys_df.iloc[0:3,1:4]

surveys_df.iloc[0:3,1:4]

surveys_df.iloc[0,3]


#subseting data using creteria
surveys_df[surveys_df.Salary==30000]
surveys_df[surveys_df.Salary!=30000]

surveys_df[surveys_df.Salary>30000]

surveys_df[surveys_df.Empno >= 204 & (surveys_df.Salary<=30000)]

surveys_df[surveys_df.Name =="akash"]

#Subsetting using methods

surveys_df[surveys_df.month.isin(['February','April','September']) & (surveys_df.Year ==2002)]



#creating new variable

import pandas as pd

dataset = pd.read_csv("data1.csv")
dataset
data=dataset.copy()
data=pd.DataFrame(data)
data['sum']=data.x1+data.x2
data
data['Mean']=(data.x1+data.x2)/2
data




