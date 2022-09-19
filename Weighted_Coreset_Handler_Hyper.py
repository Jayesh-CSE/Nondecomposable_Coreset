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


Core_Size = [175, 200]

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

	    lewis_one_vs_all_sum = 0
	    uniform_one_vs_all_sum = 0

	    #lewis_var = []
	    #uniform_var = []

	    for i in range(5):

	    	#lewis_var_trial = []
	    	#uniform_var_trial = []

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

	    		#p1 = F1_score_Weighted.F1_score(x_coreset, y_coreset, x_test, y_test, Cw_coreset, r)
	    		p1 = F1_score_Weighted.F1_score(x_coreset, y_coreset, x_train, y_train, Cw_coreset, r)
	    		lewis_weight_final, candidate_set_lewis, inter_F1_lewis = p1.Main()
	    		#p1_pred = F1_score_Weighted.F1_score(x_train, y_train, x_test, y_test, Cw_coreset, r)
	    		f1_lewis = p1.Predict(lewis_weight_final)

	    		if(max_f1 < f1_lewis):
		    		max_f1 = f1_lewis

	    	lewis_one_vs_all_sum = lewis_one_vs_all_sum + max_f1
	    	#lewis_var_trial.append(max_f1)

	    	#print("Lewis f1 score : {}".format(max_f1))

	    	uni_idx = Coresets.get_uniform(x_train[idx_minus1], y_train[idx_minus1], int(np.unique(y_train, return_counts=True)[1][0] * size_c / (np.unique(y_train, return_counts=True)[1][0] + np.unique(y_train, return_counts=True)[1][1])), seed=i)

	    	x_uni, y_uni = x_train[uni_idx], y_train[uni_idx]

	    	#Cw_core = np.ones(int(np.unique(y_train, return_counts=True)[1][0] * size_c / (np.unique(y_train, return_counts=True)[1][0] + np.unique(y_train, return_counts=True)[1][1])))*x_train.shape[0]/int(np.unique(y_train, return_counts=True)[1][0] * size_c / (np.unique(y_train, return_counts=True)[1][0] + np.unique(y_train, return_counts=True)[1][1]))

	    	Cw_core = np.ones(size_c)*x_train.shape[0]/ size_c

	    	uni_idx_plus = Coresets.get_uniform(x_train[idx_1], y_train[idx_1], int(np.unique(y_train, return_counts=True)[1][1] * size_c / (np.unique(y_train, return_counts=True)[1][0] + np.unique(y_train, return_counts=True)[1][1])), seed=i)

	    	x_uni_plus, y_uni_plus = x_train[uni_idx_plus], y_train[uni_idx_plus]

	    	#Cw_core_plus = np.ones(int(np.unique(y_train, return_counts=True)[1][1] * size_c / (np.unique(y_train, return_counts=True)[1][0] + np.unique(y_train, return_counts=True)[1][1])))*x_train.shape[0]/int(np.unique(y_train, return_counts=True)[1][1] * size_c / (np.unique(y_train, return_counts=True)[1][0] + np.unique(y_train, return_counts=True)[1][1]))

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
	    		#p3 = F1_score_Weighted.F1_score(x_coreset, y_coreset, x_test, y_test, Cw_core, r)
	    		
	    		p3 = F1_score_Weighted.F1_score(x_coreset, y_coreset, x_train, y_train, Cw_core, r)

	    		uniform_weight_final, candidate_set_uniform, inter_F1_uniform = p3.Main()

	    		#p3_pred = F1_score_Weighted.F1_score(x_train, y_train, x_test, y_test, Cw_core, r)

	    		f1_uniform = p3.Predict(uniform_weight_final)

	    		if(max_f1_uni < f1_uniform):
		    		max_f1_uni = f1_uniform
		    		best_R = r

	    	uniform_one_vs_all_sum = uniform_one_vs_all_sum + max_f1_uni

	    	#uniform_var_trial.append(max_f1_uni)

	    	#print("Uniform f1 score : {}".format(max_f1_uni))

	    	#print("Best R is : {}".format(best_R))
		    

	    print("One vs all Lewis f1 score : {}".format(lewis_one_vs_all_sum/5))
	    print("One vs all Uniform f1 score : {}".format(uniform_one_vs_all_sum/5))

	    #lewis_var.append(np.var(lewis_var_trial))
	    #uniform_var.append(np.var(uniform_var_trial))

	    lewis_one_vs_all.append(lewis_one_vs_all_sum/5)
	    uniform_one_vs_all.append(uniform_one_vs_all_sum/5)

	print("Lewis one vs all for coreset size: {} is {}".format(size_c, lewis_one_vs_all))
	print("Uniform one vs all for coreset size: {} is {}".format(size_c, uniform_one_vs_all))

	lewis_score_f1_final.append(lewis_one_vs_all)
	uniform_f1_final.append(uniform_one_vs_all)

print("Lewis Score_F1_Final: {}".format(lewis_score_f1_final))
print("Uniform_F1_Final: {}".format(uniform_f1_final))

	#lewis_score_variance.append(lewis_var)
	#uniform_variance.append(uniform_var)

	#print("Lewis score coreset size: {} , variance list: {}".format(size_c, lewis_var))
	#print("Uniform score coreset size: {} , variance list: {}".format(size_c, uniform_var))

#print("Lewis score coreset size: {} ,  final variance list: {}".format(size_c, lewis_score_variance))
#print("Uniform score coreset size: {} ,  final variance list: {}".format(size_c, uniform_variance))



		#lewis_sum = lewis_sum + (lewis_one_vs_all_sum/5)

	#lewis_score_f1.append(lewis_sum/7)	    

		    #lewis_score.append(f1_lewis)

""" #start

		    x_corr_1, y_corr_1, Cw1 = Coresets.Leverage_Coreset(x_train[idx_minus1], y_train[idx_minus1], int(np.unique(y_train, return_counts=True)[1][0] * size_c / (np.unique(y_train, return_counts=True)[1][0] + np.unique(y_train, return_counts=True)[1][1])))
		    x_corr, y_corr, Cw2 = Coresets.Leverage_Coreset(x_train[idx_1], y_train[idx_1], int(np.unique(y_train, return_counts=True)[1][1] * size_c / (np.unique(y_train, return_counts=True)[1][0] + np.unique(y_train, return_counts=True)[1][1])))
		    
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


		    
		    print("Leverage Coreset:{} ".format(np.unique(y_coreset, return_counts=True)))

		    p2 = F1_score_Weighted.F1_score(x_coreset, y_coreset, x_test, y_test, Cw_coreset)
		    leverage_weight_final, candidate_set_leverage, inter_F1_leverage = p2.Main()

		    #plt.plot(candidate_set_leverage, inter_F1_leverage , '--r', marker='o', label = 'Leverage Score')
		    #plt.xlabel('Candidate Set Size')
		    #plt.ylabel('F1-Score')
		    #plt.legend()
		    #plt.savefig('Plot_Intemediate_Weighted/Leverage_Weighted_coreset_size_{}_iteration_{}.png'.format(size_c, i))
		    #plt.show()

		    print(leverage_weight_final)
		    f1_leverage = p2.Predict(leverage_weight_final)
		    leverage_sum = leverage_sum + f1_leverage	    
		    print("Leverage Average f1 score : {}".format(f1_leverage))

		    #leverage_score.append(f1_leverage)


		    #uni_idx = Coresets.get_uniform(x_train, y_train, size_c, seed=i)
		    #x_uni, y_uni = x_train[uni_idx], y_train[uni_idx]
		    

		    uni_idx = Coresets.get_uniform(x_train[idx_minus1], y_train[idx_minus1], int(np.unique(y_train, return_counts=True)[1][0] * size_c / (np.unique(y_train, return_counts=True)[1][0] + np.unique(y_train, return_counts=True)[1][1])), seed=i)

		    x_uni, y_uni = x_train[uni_idx], y_train[uni_idx]

		    print("Uniform Coreset minus:{} ".format(np.unique(y_uni, return_counts=True)))

		    Cw_core = np.ones(int(np.unique(y_train, return_counts=True)[1][0] * size_c / (np.unique(y_train, return_counts=True)[1][0] + np.unique(y_train, return_counts=True)[1][1])))*x_train.shape[0]/int(np.unique(y_train, return_counts=True)[1][0] * size_c / (np.unique(y_train, return_counts=True)[1][0] + np.unique(y_train, return_counts=True)[1][1]))


		    uni_idx_plus = Coresets.get_uniform(x_train[idx_1], y_train[idx_1], int(np.unique(y_train, return_counts=True)[1][1] * size_c / (np.unique(y_train, return_counts=True)[1][0] + np.unique(y_train, return_counts=True)[1][1])), seed=i)

		    x_uni_plus, y_uni_plus = x_train[uni_idx_plus], y_train[uni_idx_plus]

		    print("Uniform Coreset plus:{} ".format(np.unique(y_uni, return_counts=True)))

		    Cw_core_plus = np.ones(int(np.unique(y_train, return_counts=True)[1][1] * size_c / (np.unique(y_train, return_counts=True)[1][0] + np.unique(y_train, return_counts=True)[1][1])))*x_train.shape[0]/int(np.unique(y_train, return_counts=True)[1][1] * size_c / (np.unique(y_train, return_counts=True)[1][0] + np.unique(y_train, return_counts=True)[1][1]))

		    x_coreset = np.vstack([x_uni, x_uni_plus])
		    y_coreset = np.hstack([y_uni, y_uni_plus]).flatten()
		    Cw_coreset = np.hstack([Cw_core, Cw_core_plus]).flatten()
		    
		    df = pd.DataFrame(x_coreset)
		    df['y_coreset'] = y_coreset
		    df['Cw_coreset'] = Cw_coreset
		    df = df.sample(frac = 1)
		    
		    x_coreset = df.iloc[: , :-2].values
		    y_coreset = df['y_coreset'].values
		    Cw_coreset = df['Cw_coreset'].values


		    p3 = F1_score_Weighted.F1_score(x_coreset, y_coreset, x_test, y_test, Cw_coreset)
		    uniform_weight_final, candidate_set_uniform, inter_F1_uniform = p3.Main()

		    '''
		    plt.plot(candidate_set_lewis, inter_F1_lewis , '--b', marker='o', label = 'Lewis Score')
		    plt.plot(candidate_set_leverage, inter_F1_leverage , '--g', marker='o', label = 'Leverage Score')
		    plt.plot(candidate_set_uniform, inter_F1_uniform , '--r', marker='o', label = 'Uniform Coreset')
		    plt.xlabel('Candidate Set Size')
		    plt.ylabel('F1-Score')
		    plt.legend()
		    plt.savefig('Plot_Intemediate_Weighted/Weighted_coreset_size_{}_iteration_{}.png'.format(size_c, i))
		    plt.show()

		    '''
		    print(uniform_weight_final)
		    f1_uniform = p3.Predict(uniform_weight_final)
		    uniform_sum = uniform_sum + f1_uniform
		    print("Uniform Average f1 score : {}".format(f1_uniform))

		    #uniform.append(f1_uniform)

#end """

		#lewis_score_f1.append(lewis_sum/5)
		#leverage_score_f1.append(leverage_sum/5)
		#uniform_f1.append(uniform_sum/5)
	
	

#plt.plot(Core_Size, lewis_score_f1, '--b', marker='o', label = 'lewis score')
#plt.plot(Core_Size, leverage_score_f1, '--g', marker='o', label = 'leverage score')
#plt.plot(Core_Size, uniform_f1, '--r', marker='o', label = 'uniform coreset')
#plt.xlabel('Coreset Size')
#plt.ylabel('F1-Score')
#plt.legend()
#plt.savefig('Plot_Intemediate_Weighted/20C_Weighted_Final.png')
#plt.show()