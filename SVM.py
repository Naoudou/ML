# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 17:44:58 2023

@author: pcsf28
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#from sklearn.naive_bayes import GaussianNB

#importing the data
data = pd.read_csv('kyphosis.csv')

#setting up the data
x = data.iloc[:,[1,2,3]]
y = data.iloc[:,0]


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)


#training the data
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)



#training the classifier

from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(x_train, y_train)

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
cm1 = confusion_matrix(y_test, y_pred, labels=["absent","present"])
print(cm1)

df_confusion = pd.crosstab(y_test, y_pred)
print(df_confusion)

df_confusion1=pd.crosstab(y_test, y_pred, rownames=["actual"],colnames=["Predicted"],margins=True)
print(df_confusion1)

sensitivity1= cm1[1,1]/(cm1[1,0] + cm1[1,1])
print("Sensitivity:", sensitivity1)

specificity1= cm1[0,0]/(cm1[0,0] + cm1[0,1])
print("Specificity:", specificity1)

