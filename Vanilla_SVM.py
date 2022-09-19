import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import metrics
import sklearn
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

from sklearn import svm


from scipy.optimize import minimize
from numpy import linalg as LA

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

#CovType Dataset load

from sklearn.datasets import fetch_covtype
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer


x_train = np.load("F1_2000_x_train.npy")

x_test = np.load("F1_2000_x_test.npy")

y_train = np.load("F1_2000_y_train.npy")

y_test = np.load("F1_2000_y_test.npy")

print("Stratified Class distribution : {}".format(np.unique(y_train, return_counts=True)))

mm = make_pipeline(MinMaxScaler(), Normalizer())
x_train = mm.fit_transform(x_train)
x_test = mm.transform(x_test)

ones_train = np.ones(x_train.shape[0])
ones_test = np.ones(x_test.shape[0])

ones_train.shape = (x_train.shape[0],1)
ones_test.shape = (x_test.shape[0],1)

x_train = np.append(x_train, ones_train, axis = 1)
x_test = np.append(x_test, ones_test, axis = 1)


def f1_score_pr(precision, recall):

    # F1 = 2 * (precision * recall) / (precision + recall)

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


clf = svm.SVC(kernel='linear', probability = True) # Linear Kernel

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

prob_test = clf.predict_proba(x_test)

max_f1 = 0
best_th = 0.2

f1_list = []
threshold = [0.2, 0.3 , 0.4, 0.5, 0.6 , 0.7, 0.8, 0.9] 

for th in threshold:
    
    y_prediction = (prob_test[:,1] > th)
    print("threshold : {}".format(th))
    y_prediction = np.array([int(x) for x in y_prediction])
    y_prediction[y_prediction == 0] = -1
  
    precision = metrics.precision_score(y_test, y_prediction, zero_division = 1)
    recall = metrics.recall_score(y_test, y_prediction, zero_division = 1)
    f1 = f1_score_pr(precision, recall)
    print("f1 score : {}".format(f1))
    f1_list.append(f1)

    if(max_f1 < f1):
        max_f1 = f1 
        best_th = th
        best_y_pred = np.copy(y_prediction)

print("Best_th : {}".format(best_th))
print("Best_y_hat : {}".format(best_y_pred))
print("Best f1 score : {}".format(max_f1))
print("Label distribution for the test dataset: {}".format(np.unique(y_test, return_counts=True)))
print("Label distribution for predicted samples: {}".format(np.unique(best_y_pred, return_counts=True)))
