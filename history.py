
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

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
x=data.data
y=data.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', max_iter=10000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_pred_prob = model.predict_proba(x_test)[:,1]
#ROC CURVE
from sklearn.metrics import roc_curve, roc_auc_score
fpr,tpr,thresholds = roc_curve(y_test, y_pred)

#compute the ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_prob)
roc_auc
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],'--')
plt.xlabel('False positive rate')
plt.ylabel('TRUE positive rate')
plt.title('roc cure')
plt.show()
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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv('kyphosis.csv')


x = data.iloc[:,[1,2,3]]
y = data.iloc[:,0]


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)


#training the data
from sklearn.linear_model import LogisticRegression
clf= LogisticRegression(random_state=0)
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
cm1=confusion_matrix(y_test, y_pred, labels=["absent","present"])
print(cm1)
df_confusion=pd.crosstab(y_test, y_pred)
print(df_confusion)

df_confusion1=pd.crosstab(y_test, y_pred, rownames=["actual"],colnames=["Predicted"],margins=True)
print(df_confusion1)


import matplotlib.pyplot as plt
import pandas as pd



df=pd.read_csv("kmeans.csv")
df.head()

import os
os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.cluster import KMeans

sum_of_squared_distances = []
K = range(1, 15)

for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df)
    sum_of_squared_distances.append(km.inertia_)


plt.plot(K, sum_of_squared_distances,'bx-')
plt.xlabel('k')
plt.ylabel("sum_of_squared_distances")
plt.title("elbow method for optimal K")
plt.show()

import matplotlib.pyplot as plt
import pandas as pd



df=pd.read_csv("kmeans.csv")
df.head()

import os
os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.cluster import KMeans

sum_of_squared_distances = []
K = range(1, 15)

for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df)
    sum_of_squared_distances.append(km.inertia_)


plt.plot(K, sum_of_squared_distances,'bx-')
plt.xlabel('k')
plt.ylabel("sum_of_squared_distances")
plt.title("elbow method for optimal K")
plt.show()
plt.plot(K, sum_of_squared_distances,'bx-')
plt.xlabel('k')
plt.ylabel("sum_of_squared_distances")
plt.title("elbow method for optimal K")
plt.show()
plt.plot(K, sum_of_squared_distances,'bx-')
plt.xlabel('k')
plt.ylabel("sum_of_squared_distances")
plt.title("elbow method for optimal K")
plt.show()
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
y = dataset.iloc[:,1:1]
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,1:1]
y = dataset.iloc[:,1]
x = dataset.iloc[:,0]
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
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

import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_Customers.csv")
x= dataset.iloc[:,[3,4]].values
x

#using the dendrogram
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x, method="ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()

#Filtering hierachicaly clustering to the dataset

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean",linkage='ward')
y_hc = hc.fit_predict(x)

dataset['cluster']=y_hc
dataset.head()

dataset


df1 = dataset.sort_values(["cluster"])
df1
print(df1)


zero_member = dataset[dataset.cluster == 0]
print(zero_member)

first_member= dataset[dataset.cluster == 1]
print(first_member)

second_member= dataset[dataset.cluster == 2]
print(second_member)

third_member= dataset[dataset.cluster == 3]
print(third_member)


#visualization

plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s= 100,c = 'red', label ='cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s= 100,c = 'blue', label ='cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s= 100,c = 'green', label ='cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s= 100,c = 'cyan', label ='cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s= 100,c = 'magenta', label ='cluster 5')



plt.title("clusters of customers")
plt.xlabel("annual Income (k$)")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()

"""
Created on Wed Jul 26 14:33:27 2023

@author: pcsf28
"""


import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_Customers.csv")
x= dataset.iloc[:,[3,4]].values
x

#using the dendrogram
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x, method="ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()

#Filtering hierachicaly clustering to the dataset

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean",linkage='ward')
y_hc = hc.fit_predict(x)

dataset['cluster']=y_hc
dataset.head()

dataset


df1 = dataset.sort_values(["cluster"])
df1
print(df1)


zero_member = dataset[dataset.cluster == 0]
print(zero_member)

first_member= dataset[dataset.cluster == 1]
print(first_member)

second_member= dataset[dataset.cluster == 2]
print(second_member)

third_member= dataset[dataset.cluster == 3]
print(third_member)


#visualization

plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s= 100,c = 'red', label ='cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s= 100,c = 'blue', label ='cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s= 100,c = 'green', label ='cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s= 100,c = 'cyan', label ='cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s= 100,c = 'magenta', label ='cluster 5')



plt.title("clusters of customers")
plt.xlabel("annual Income (k$)")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()

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
print(regressor.intercept_)
print(regressor.coef_)
import matplotlib.pyplot as plt
import pandas as pd



df=pd.read_csv("kmeans.csv")
df.head()

import os
os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.cluster import KMeans

sum_of_squared_distances = []
K = range(1, 15)

for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df)
    sum_of_squared_distances.append(km.inertia_)


plt.plot(K, sum_of_squared_distances,'bx-')
plt.xlabel('k')
plt.ylabel("sum_of_squared_distances")
plt.title("elbow method for optimal K")
plt.show()

"""
Created on Thu Jul 27 15:50:33 2023

@author: pcsf28
"""



import matplotlib.pyplot as plt
import pandas as pd



df=pd.read_csv("kmeans.csv")
df.head()

import os
os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.cluster import KMeans

sum_of_squared_distances = []
K = range(1, 15)

for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df)
    sum_of_squared_distances.append(km.inertia_)


plt.plot(K, sum_of_squared_distances,'bx-')
plt.xlabel('k')
plt.ylabel("sum_of_squared_distances")
plt.title("elbow method for optimal K")
plt.show()

## ---(Tue Aug 15 17:39:58 2023)---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv('kyphosis.csv')


x = data.iloc[:,[1,2,3]]
y = data.iloc[:,0]


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)


#training the data
from sklearn.linear_model import LogisticRegression
clf= LogisticRegression(random_state=0)
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
cm1=confusion_matrix(y_test, y_pred, labels=["absent","present"])
print(cm1)
df_confusion=pd.crosstab(y_test, y_pred)
print(df_confusion)

df_confusion1=pd.crosstab(y_test, y_pred, rownames=["actual"],colnames=["Predicted"],margins=True)
print(df_confusion1)

"""
Created on Tue Jul 25 15:55:25 2023

@author: pcsf28
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv('kyphosis.csv')


x = data.iloc[:,[1,2,3]]
y = data.iloc[:,0]


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)


#training the data
from sklearn.linear_model import LogisticRegression
clf= LogisticRegression(random_state=0)
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
cm1=confusion_matrix(y_test, y_pred, labels=["absent","present"])
print(cm1)
df_confusion=pd.crosstab(y_test, y_pred)
print(df_confusion)

df_confusion1=pd.crosstab(y_test, y_pred, rownames=["actual"],colnames=["Predicted"],margins=True)
print(df_confusion1)
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

from sklearn.metrics import confusion_matrix

cm1=confusion_matrix(y_test, y_pred, labels=["absent","present"])
print(cm1)
df_confusion=pd.crosstab(y_test, y_pred)
print(df_confusion)
cm1=confusion_matrix(y_test, y_pred, labels=["absent","present"])
print(cm1)
df_confusion=pd.crosstab(y_test, y_pred)
print(df_confusion)
print(df_confusion)

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
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection, svm
x, y = datasets.make_classification(random_state=0)
x_train, x_test, y_train, y_test =model_selection.train_test_split(x,y,test_size=0.25,random_state=0)
clf = svm.SVC(random_state=0)
clf.fit(x_train, y_train)
metrics.plot_roc_curve(clf,x_test,y_test)
plt.show()

import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection, svm
x, y = datasets.make_classification(random_state=0)
x_train, x_test, y_train, y_test =model_selection.train_test_split(x,y,test_size=0.25,random_state=0)
clf = svm.SVC(random_state=0)
clf.fit(x_train, y_train)
metrics.plot_roc_curve(clf,x_test,y_test)
plt.show()
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=2,random_state=0 )
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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