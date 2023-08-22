# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:53:15 2023

@author: pcsf28
"""

import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection, svm
x, y = datasets.make_classification(random_state=0)
x_train, x_test, y_train, y_test =model_selection.train_test_split(x,y,test_size=0.25,random_state=0)
clf = svm.SVC(random_state=0)
clf.fit(x_train, y_train)
metrics.plot_roc_curve(clf,x_test,y_test)
plt.show()
