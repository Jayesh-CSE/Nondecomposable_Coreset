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

import Joachims_Weighted
import Coresets



R = [0.01 , 0.1 , 1 , 2 , 4]

uniform_f1_final = []
lewis_score_f1_final = []


Core_Size = [25, 50, 100, 150, 175, 200]


#Dataset load and preprocessing

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
print("Test class distribution: {}".format(np.unique(y_test, return_counts=True)))

mm = make_pipeline(MinMaxScaler(), Normalizer())
x_train = mm.fit_transform(x_train)
x_test = mm.transform(x_test)

ones_train = np.ones(x_train.shape[0])
ones_test = np.ones(x_test.shape[0])

ones_train.shape = (x_train.shape[0],1)
ones_test.shape = (x_test.shape[0],1)

x_train = np.append(x_train, ones_train, axis = 1)
x_test = np.append(x_test, ones_test, axis = 1)

idx_minus1 = np.where(y_train!= 1)[0]

idx_1 = np.where(y_train != -1)[0]


for size_c in Core_Size:

	lewis_one_vs_all = []
	uniform_one_vs_all = []


	for i in range(1,2):

	    lewis_one_vs_all_sum = 0
	    uniform_one_vs_all_sum = 0

	    for i in range(3):

	    	print("Coreset Size : {}".format(size_c))

	    	x_corr_1, y_corr_1, Cw1 = Coresets.Lewis_Coreset(x_train[idx_minus1], y_train[idx_minus1], int(np.unique(y_train, return_counts=True)[1][0] * size_c / (np.unique(y_train, return_counts=True)[1][0] + np.unique(y_train, return_counts=True)[1][1])))
	    	x_corr, y_corr, Cw2 = Coresets.Lewis_Coreset(x_train[idx_1], y_train[idx_1], int(np.unique(y_train, return_counts=True)[1][1] * size_c / (np.unique(y_train, return_counts=True)[1][0] + np.unique(y_train, return_counts=True)[1][1])))

	    	x_coreset = np.vstack([x_corr_1, x_corr])

	    	y_coreset = np.hstack([y_corr_1, y_corr]).flatten()

	    	Cw_coreset = np.hstack([Cw1, Cw2]).flatten()

	    	df = pd.DataFrame(x_coreset)
	    	df['y_coreset'] = y_coreset
	    	df['Cw_coreset'] = Cw_coreset
	    	df = df.sample(frac = 1)
	    	x_coreset = df.iloc[: , :-2].values
	    	y_coreset = df['y_coreset'].values
	    	Cw_coreset = df['Cw_coreset'].values

	    	print("Lewis Coreset:{} ".format(np.unique(y_coreset, return_counts=True)))

	    	max_f1 = 0

	    	for r in R:

	    		p1 = Joachims_Weighted.F1_score(x_coreset, y_coreset, x_train, y_train, Cw_coreset, r)
	    		lewis_weight_final, candidate_set_lewis, inter_F1_lewis = p1.Main()
	    		f1_lewis = p1.Predict(lewis_weight_final)

	    		if(max_f1 < f1_lewis):
		    		max_f1 = f1_lewis

	    	lewis_one_vs_all_sum = lewis_one_vs_all_sum + max_f1
	    	

	    	print("Lewis f1 score : {}".format(max_f1))

	    	
	    	uni_idx = Coresets.get_uniform(x_train[idx_minus1], y_train[idx_minus1], int(491.0 * size_c /1000.0), seed=i)

	    	x_uni, y_uni = x_train[uni_idx], y_train[uni_idx]

	    	#Cw_core = np.ones(int(491.0 * size_c /1000.0))*x_train.shape[0]/(int(491.0 * size_c /1000.0))

	    	Cw_core = np.ones(size_c)*x_train.shape[0]/ size_c

	    	uni_idx_plus = Coresets.get_uniform(x_train[idx_1], y_train[idx_1], int(509 * size_c /1000), seed=i)

	    	x_uni_plus, y_uni_plus = x_train[uni_idx_plus], y_train[uni_idx_plus]

	    	#Cw_core_plus = np.ones(int(509 * size_c /1000))*x_train.shape[0]/(int(509 * size_c /1000))

	    	x_coreset = np.vstack([x_uni, x_uni_plus])
	    	y_coreset = np.hstack([y_uni, y_uni_plus]).flatten()
	    	#Cw_coreset = np.hstack([Cw_core, Cw_core_plus]).flatten()

	    	df = pd.DataFrame(x_coreset)
	    	df['y_coreset'] = y_coreset
	    	#df['Cw_coreset'] = Cw_coreset
	    	df = df.sample(frac = 1)

	    	x_coreset = df.iloc[: , :-1].values
	    	y_coreset = df['y_coreset'].values
	    	#Cw_coreset = df['Cw_coreset'].values

	    	print("Uniform Coreset :{} ".format(np.unique(y_coreset, return_counts=True)))

	    	max_f1_uni = 0

	    	best_R = 0

	    	for r in R:
	    		p3 = Joachims_Weighted.F1_score(x_coreset, y_coreset, x_train, y_train, Cw_core, r)
	    		uniform_weight_final, candidate_set_uniform, inter_F1_uniform = p3.Main()

	    		f1_uniform = p3.Predict(uniform_weight_final)

	    		if(max_f1_uni < f1_uniform):
		    		max_f1_uni = f1_uniform
		    		best_R = r

	    	uniform_one_vs_all_sum = uniform_one_vs_all_sum + max_f1_uni

	    	
	    	print("Uniform f1 score : {}".format(max_f1_uni))

	    	#print("Best R is : {}".format(best_R))
		    

	    print("One vs all Lewis f1 score : {}".format(lewis_one_vs_all_sum/3))
	    print("One vs all Uniform f1 score : {}".format(uniform_one_vs_all_sum/3))

	    
	    lewis_one_vs_all.append(lewis_one_vs_all_sum/3)
	    uniform_one_vs_all.append(uniform_one_vs_all_sum/3)

	print("Lewis one vs all for coreset size: {} is {}".format(size_c, lewis_one_vs_all))
	print("Uniform one vs all for coreset size: {} is {}".format(size_c, uniform_one_vs_all))

	lewis_score_f1_final.append(lewis_one_vs_all)
	uniform_f1_final.append(uniform_one_vs_all)

print("Lewis f1 score final : {}".format(lewis_score_f1_final))
print("Uniform f1 score final: {}".format(uniform_f1_final))
