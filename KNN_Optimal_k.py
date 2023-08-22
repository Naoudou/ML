# -*- coding: utf-8 -*-
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



