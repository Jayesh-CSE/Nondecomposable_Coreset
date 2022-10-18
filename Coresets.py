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

#Compute leverage score

def lev_score(x):
    U, S, V = np.linalg.svd(x.astype(float) ,full_matrices=False)
    U1 = U[:x.shape[0], :]
    proba = np.square(U1).sum(axis=1)/np.square(U1).sum(axis=1).sum()
    return proba



def Leverage_Coreset(x_train, y_train, coreset_size):
    
    prob = lev_score(x_train)
    prob.shape = (1, x_train.shape[0])
    
    #np.random.seed(0)
    
    core_idx = np.random.choice(len(x_train), size=coreset_size, replace=True, p=prob[0])  #with replacement.....

    #core_idx = np.random.choice(len(x_train), size=coreset_size, p=prob[0])
    
    Cw = 1/prob[0][core_idx]
    
    Cw.shape = (1,coreset_size)

    #X_core = Cw.T * x_train[core_idx]
    X_core = x_train[core_idx]
    y_core = y_train[core_idx]
    
    return X_core , y_core , Cw 



#Code for Compute W^(-1/2) and reset off diagonal entry to zero

def w_power(w, p= -0.5):
    with np.errstate(divide='ignore'):
        demo2 = np.float_power(w, p)
    demo2[np.isinf(demo2)] = 0
    demo2[np.isnan(demo2)] = 0
    return demo2
	
#Code for Lewis Iterate

def lewis_iterate(w, x):
    for i in range(20):
        lev_sc = lev_score(w_power(w).T * x) 
        w = w_power(w_power(w, -1)[0].T * lev_sc , 0.5)
        w.shape = (1,w.shape[0])
    return w
	
	
def Lewis_Coreset(x_train, y_train, coreset_size):
    
    full_data = x_train.shape[0]

    #print(full_data)
    
    w_lewis = np.ones(full_data)
    w_lewis.shape = (1,full_data)

    p_i = lewis_iterate(w_lewis , x_train)
    prob = p_i/p_i.sum()

    #print(prob.min())
    #print(prob.max())
    #print(prob.sum())

    prob.shape = (1, x_train.shape[0])
    
    #np.random.seed(0)
    
    #core_idx = np.random.choice(len(x_train), size=N_core, replace=True, p=prob[0])  with replacement.....

    core_idx = np.random.choice(len(x_train), size=coreset_size, p=prob[0])
    
    Cw = 1/prob[0][core_idx]

    #print(Cw.sum())
    #print(Cw.min())
    #print(Cw.max())
    
    Cw.shape = (1,coreset_size)

    #X_core = Cw.T * x_train[core_idx]
    X_core = x_train[core_idx]
    y_core = y_train[core_idx]
    
    return X_core , y_core , Cw 
     
	 
def get_uniform(x, y, n, seed):
    np.random.seed(seed)
    proba = np.ones(x.shape[0])/x.shape[0]
    idx = np.random.choice(x.shape[0], size=n, p=proba)
    #return idx , proba
    return idx
