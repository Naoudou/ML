# -*- coding: utf-8 -*-
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


