import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import metrics
import sklearn


import warnings
warnings.filterwarnings("ignore")

from sklearn import svm

from scipy.optimize import minimize
from numpy import linalg as LA

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from sklearn.datasets import fetch_covtype
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer

import F1_score_hyperparam

R = [0.01 , 0.1 , 1 , 2 , 4]

f1_score_fulldata = []


X = np.load("x_data.npy")
y = np.load("y_data.npy")

y[y == 1] = -1
y[y == 3] = -1
y[y == 6] = -1
y[y == 7] = -1

y[y == 2] = 1
y[y == 4] = 1
y[y == 5] = 1

print("Original Class distribution: {}".format(np.unique(y, return_counts=True)))

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=1000, test_size=2000, stratify = y, random_state=108)

print("Train class distribution: {}".format(np.unique(y_train, return_counts=True)))

#y_train.shape = (y_train.shape[0], 1)
#data = np.hstack((x_train, y_train))
#df = pd.DataFrame(data)
#data_main = df.groupby(54).sample(n=500, replace = True, random_state=1).to_numpy()
#x_train = data_main[:, :-1]
#y_train = data_main[:, -1]
#print(np.unique(y_train, return_counts=True))


mm = make_pipeline(MinMaxScaler(), Normalizer())
x_train = mm.fit_transform(x_train)
x_test = mm.transform(x_test)

ones_train = np.ones(x_train.shape[0])
ones_test = np.ones(x_test.shape[0])

ones_train.shape = (x_train.shape[0],1)
ones_test.shape = (x_test.shape[0],1)

x_train = np.append(x_train, ones_train, axis = 1)
x_test = np.append(x_test, ones_test, axis = 1)

sum_one_vs_rest = 0

for i in range(5):

    max_f1 = 0

    for r in R:

        p1 = F1_score_hyperparam.F1_score(x_train, y_train, x_test, y_test, r)

        weight_final, candidate_set, inter_F1 = p1.Main()

        print("f1 score : {}".format(p1.Predict(weight_final)))

        print("Test samples label distribution: {}".format(np.unique(y_test, return_counts=True)))

        f1_r = p1.Predict(weight_final)

        if(max_f1 < f1_r):
        	max_f1 = f1_r

    sum_one_vs_rest = sum_one_vs_rest + max_f1
    print("f1 score for {} th iteration is: {}".format(i, max_f1))

f1_score_fulldata.append(sum_one_vs_rest/5)

print("F1 score for full data is : {}".format(f1_score_fulldata)) 
    


