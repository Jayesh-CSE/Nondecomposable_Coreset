#Import all packages 

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

import F1_score_Weighted
import Coresets


#R = [-64, -32, -16, -8, -4 , -2, 1 , 2, 4, 8, 16, 32, 64]

R = [0.01 , 0.1 , 1 , 2 , 4]

#leverage_score_f1_final = []
uniform_f1_final = []
lewis_score_f1_final = []


Core_Size = [175]

#Core_Size = [100, 150, 175, 200]


for size_c in Core_Size:

	#lewis_score_variance = []
	#leverage_score = []
	#uniform_variance = []


	lewis_one_vs_all = []
	#leverage_one_vs_all = []
	uniform_one_vs_all = []


	for i in range(1,8):

	    #X, y = fetch_covtype(return_X_y=True)

	    X = np.load("x_data.npy")
	    y = np.load("y_data.npy")

	    y[y != i] = -1
	    y[y == i] = 1
	    
	    #print("Original Class distribution: {}".format(np.unique(y, return_counts=True)))
	    
	    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=1000, test_size=2000, stratify = y, random_state=108)

	    print("Train class distribution: {}".format(np.unique(y_train, return_counts=True)))
