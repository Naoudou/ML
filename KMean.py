# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:29:35 2023

@author: pcsf28
"""

import pandas as pd



df=pd.read_csv("kmeans.csv")
df.head()


from sklearn.cluster import KMeans
km = KMeans(n_clusters=4, init="k-means++", n_init=10)

km.fit(df)

y_hc= km.fit_predict(df)
y_hc

df["cluster"]= y_hc
df.head


df

X=df
df1 = df.sort_values(["cluster"])
df1
print(df1)


zero_member= df[df.cluster == 0]
print(zero_member)

first_member= df[df.cluster == 1]
print(first_member)

second_member= df[df.cluster == 2]
print(second_member)

third_member= df[df.cluster == 3]
print(third_member)
