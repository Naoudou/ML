# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:02:55 2023

@author: pcsf28
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('kyphosis.csv')


x = data.iloc[:,[1,2,3]]
y = data.iloc[:,0]


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)


#training the data
from sklearn.linear_model import LogisticRegression
clf= LogisticRegression()
clf= clf.fit(x_train, y_train)


#predicting
y_pred = clf.predict(x_test)
y_pred
##df= pd.DataFrame({"Actual":y_test,"Predict":y_pred})
##df
#testin accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy
print(str(accuracy*100)+"% Accuracy")
    
##Making the confusion matrix
cm1=confusion_matrix(y_test, y_pred, labels=["Absent","Present"])
print(cm1)
df_confusion=pd.crosstab(y_test, y_pred)
print(df_confusion)

df_confusion1=pd.crosstab(y_test, y_pred, rownames=["actual"],colnames=["Predicted"],margins=True)
print(df_confusion1)

