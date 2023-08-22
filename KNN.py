

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:30:51 2023

@author: pcsf28
"""
import pandas as pd





data = pd.read_csv('kyphosis.csv')

x = data.iloc[:,[1,2,3]]
y = data.iloc[:,0]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)


#training the data
from sklearn.neighbors import KNeighborsClassifier
clf= KNeighborsClassifier(n_neighbors = 13)
clf= clf.fit(x_train, y_train)


#predicting
y_pred = clf.predict(x_test)
y_pred
##df= pd.DataFrame({"Actual":y_test,"Predict":y_pred})
##df
#testin accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy
print(str(accuracy*100)+"% Accuracy")

##Making the confusion matrix
from sklearn.metrics import confusion_matrix

cm1=confusion_matrix(y_test, y_pred, labels=["absent","present"])
print(cm1)
df_confusion=pd.crosstab(y_test, y_pred)
print(df_confusion)

df_confusion1=pd.crosstab(y_test, y_pred, rownames=["actual"],colnames=["predicted"],margins=True)
print(df_confusion1)

