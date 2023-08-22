# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:50:33 2023

@author: pcsf28
"""

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
plt.shox()
