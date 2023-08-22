# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 17:44:30 2023

@author: pcsf28
"""

import numpy as np
import pandas as pd


dataset = pd.read_csv('sales.csv')
dataset.head()

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,1]

#spliting data into the training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1.0/3,random_state=0)

dataset.head()

#training the data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#regressor = regressor.fit(x_train, y_train)
regressor = regressor.fit(x_train, y_train)


#print the coeficient
print(regressor.intercept_)
print(regressor.coef_)


#predicting the Test set result

y_pred = regressor.predict(x_test)
y_pred


df= pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
df

#testin accuracy
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

from sklearn import metrics
print("Mean Absolute Error", metrics.mean_absolute_error(y_test,y_pred))
print("Mean Squared Error", metrics.mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

      

 