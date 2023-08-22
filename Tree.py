# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:25:14 2023

@author: pcsf28
"""


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv('kyphosis.csv')


x = data.iloc[:,[1,2,3]]
y = data.iloc[:,0]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)


#training the data
from sklearn import tree
clf= tree.DecisionTreeClassifier(criterion="gini",min_samples_split=30,splitter="best")
clf= clf.fit(x_train, y_train)

#predicting
y_pred = clf.predict(x_test)
y_pred
df= pd.DataFrame({"Actual":y_test,"Predict":y_pred})
df
#testin accuracy
accuracy = accuracy_score(y_test, y_pred)
#print ("accuracy:",accuracy)
print(str(accuracy*100)+"% Accuracy")
cm1=confusion_matrix(y_test, y_pred, labels=["absent","present"])
print(cm1)
df_confusion=pd.crosstab(y_test, y_pred)
print(df_confusion)

df_confusion1=pd.crosstab(y_test, y_pred, rownames=["actual"],colnames=["predicted"],margins=True)
print(df_confusion1)

'''total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print("Accuracy:",accuracy)'''


sensitivity1= cm1[1,1]/(cm1[1,0] + cm1[1,1])
print("Sensitivity:", sensitivity1)

specificity1= cm1[0,0]/(cm1[0,0] + cm1[0,1])
print("Specificity:", specificity1)
