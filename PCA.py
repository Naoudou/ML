# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 16:46:45 2023

@author: pcsf28
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 15:53:41 2023

@author: pcsf28
"""

import pandas as pd

dataset = pd.read_csv('iris.csv')
dataset.head()


x=dataset.iloc[:,0:4].values
y=dataset.iloc[:,4].values

#spliting the dataset into training set ad Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler

sc =StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
from sklearn.decomposition import PCA


pca =PCA()
x_train= pca.fit_transform(x_train)
x_test= pca.fit_transform(x_test)
explained_variance = pca.explained_variance_ratio_

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=2,random_state=0 )
classifier.fit(x_train, y_train)

#predicting the test set resulte
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
accuracy



