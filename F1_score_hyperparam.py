import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import metrics
import sklearn

from sklearn import model_selection

import warnings
warnings.filterwarnings("ignore")

from sklearn import svm

from scipy.optimize import minimize
from numpy import linalg as LA



class F1_score:

    def __init__(self, x_train, y_train, x_test, y_test, R):
        
        self.epsilon = 0.01
        self.C = []
        self.loss_fun = []
        self.R = R
        self.x_train_n = x_train
        self.y_train_n = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.weight = np.random.rand(x_train.shape[1])

    def f1_score_new(self, a, b, c, d):

        tp = a
        fp = b
        fn = c

        pr_div = tp+fp
        re_div = tp+fn

        if(pr_div == 0):
            pr_div = 1

        if(re_div == 0):
            re_div = 1
      
        precision = tp/pr_div
        recall = tp/re_div

        denominator = precision + recall

        if(denominator == 0):
            denominator = 1

        f1 = 2 * (precision * recall) / denominator

        return f1
    
    def psi(self, X, y_prime):  

        l = np.zeros(X.shape[1])
        for i in range(X.shape[0]):
            l = np.add(l,y_prime[i] * X[i])
        return l

    #Function for algorithm 2

    def fun_argmax_y(self, X, Y, W):
        
        ind_y_pos = []   
        ind_y_neg = [] 

        for i in range(Y.shape[0]):    
            if(Y[i] == 1):
                ind_y_pos.append((i,np.dot(W, X[i])))
            elif(Y[i] == -1):
                ind_y_neg.append((i,np.dot(W, X[i])))

        ind_y_pos.sort(key=lambda ind_y_pos: -ind_y_pos[1])
        ind_y_neg.sort(key=lambda ind_y_neg: ind_y_neg[1])

        num_positive = len(ind_y_pos)
        num_negative = len(ind_y_neg)

        counter_v = 0

        y_bar_star = [None] * (num_positive + num_negative)

        psi_sum = np.zeros(X.shape[1])

        
        for a in range(0,num_positive+1):   # 0 and 1 and 2
            
            c = num_positive - a

            if(a == 0):
                
                y_prime = [None] * (num_positive + num_negative)

                for i in ind_y_pos:
                    y_prime[i[0]] = -1
            else:
            
                #c = num_positive - a         # 2 and 1 and 0

                psi_sum = np.subtract(psi_sum, y_prime[ind_y_pos[a-1][0]] * X[ind_y_pos[a-1][0]])
              
                y_prime[ind_y_pos[a-1][0]] = 1

                psi_sum = np.add(psi_sum, y_prime[ind_y_pos[a-1][0]] * X[ind_y_pos[a-1][0]])

            for d in range(0, num_negative+1): #0 and 1 and 2

                b = num_negative - d

                if(d == 0):
                    
                    if(a == 0 and d == 0):

                        for j in ind_y_neg:
                            y_prime[j[0]] = 1

                        for i in range(X.shape[0]):
                            psi_sum = np.add(psi_sum,y_prime[i] * X[i])

                    else:

                        for j in ind_y_neg:

                            psi_sum = np.subtract(psi_sum, y_prime[j[0]] * X[j[0]])

                            y_prime[j[0]] = 1

                            psi_sum = np.add(psi_sum, y_prime[j[0]] * X[j[0]])

                else:

                    #b = num_negative - d       # 2 and 1 and 0

                    psi_sum = np.subtract(psi_sum, y_prime[ind_y_neg[b-1][0]] * X[ind_y_neg[b-1][0]])

                    y_prime[ind_y_neg[b-1][0]] = -1

                    psi_sum = np.add(psi_sum, y_prime[ind_y_neg[b-1][0]] * X[ind_y_neg[b-1][0]])

                
                #print("psi_sum: {}".format(psi_sum))

                loss_delta = self.f1_score_new(a,b,c,d)
                
                v = loss_delta + np.dot(W, psi_sum)


                if(counter_v == 0):
                    max_v = v
                    y_bar_star = y_prime[:]
                    loss_val = loss_delta
                    counter_v = 1
                else:
                    if(max_v <= v):
                        max_v = v
                        y_bar_star = y_prime[:]
                        loss_val = loss_delta
        return y_bar_star, loss_val

    def ROCA_argmax(self, X, Y, W):   

        S = []
        positive = 0
        negative = 0
        
        for i in range(Y.shape[0]):
            if(Y[i] == 1):
                S.append((i, -0.25 + np.dot(W, X[i])))
                positive = positive + 1
            elif(Y[i] == -1):
                S.append((i, 0.25 + np.dot(W, X[i])))
                negative = negative + 1

        #print(S)       
        
        S.sort(key=lambda S: S[1])

        #print(S)
        
        C = [None] * (Y.shape[0])
        R = [] 
        
        s_p = positive 
        
        s_n = 0
        
        for i in S:
            R.append(i[0])
            
        for i in range(Y.shape[0]):
            
            if(Y[R[i]] > 0):
            
                C[R[i]] = negative - 2 * s_n
                s_p = s_p - 1

            else:
            
                C[R[i]] = -positive + 2 * s_p
                s_n = s_n + 1

        loss_val = self.ROCA_loss(Y, C)

        #y_bar_star = np.sign(C)
        
        return C , loss_val
            

    def ROCA_loss(self, Y, C_Prime):

        positive = 0
        negative = 0
        
        for i in range(Y.shape[0]):
            if(Y[i] == 1):
                positive = positive + 1
            elif(Y[i] == -1):
                negative = negative + 1

        C = [None] * (Y.shape[0])

        for i in range(Y.shape[0]):
            
            if(Y[i] == 1):
            
                C[i] = negative

            else:
            
                C[i] = -positive

        loss = np.sum(Y * np.subtract(C, C_Prime))/2
                
        return loss

        
    def objective(self, A):

        W = A[:-1]
        phi = A[-1]
        return np.square(LA.norm(W, 2)) * 0.5 +  self.R * phi

    def preprocess_constraint(self, A):
    
        W = A[:-1]
        phi = A[-1]
        l = len(self.C)
        psi_original_data = self.psi(self.x_train_n, self.y_train_n)
        Mat = []
        B = self.loss_fun

        param = np.append(W, [phi])

        for i in range(l):
            Mat.append(psi_original_data - self.psi(self.x_train_n, self.C[i]))
            Mat[i] = np.append(Mat[i], [-1])

        cond = np.dot(Mat, param) - B
        return cond

    def constraint(self, A):
        return self.preprocess_constraint(A)

    def SVM_Multi(self, weight, phi):
    
        weight_phi = np.append(weight, [phi])
        
        A = [weight_phi]

        constraint_dict = ({'type':'ineq', 'fun': self.constraint})
        
        b_phi = (0, np.inf)
        b_weight = (-np.inf, np.inf)

        demo = [(b_weight)]*(self.x_train_n.shape[1])
        demo.append(b_phi)
        bnds = tuple(demo)

        sol = minimize(self.objective, A[0] , method = 'SLSQP', bounds=bnds, constraints = constraint_dict, options={'maxiter':500})
        #sol = minimize(self.objective, A[0] , method = 'SLSQP', bounds=bnds, constraints = constraint_dict)

        return sol.x[:-1] , sol.x[-1]

    def f1_score_pr(self, precision, recall):
        
        denom = precision+ recall
        
        if(denom == 0):
            denom = 1
        f1 = 2 * (precision * recall) / (denom)

        return f1

    def Main(self):

        psi_original_data = self.psi(self.x_train_n, self.y_train_n)
        
        counter_update = 1 
        
        count = 0

        candidate_set = []
        F1_score_inter = []

        while(counter_update and count < 20):
        #while(counter_update):

            counter_update = 0 
            count = count + 1
            
            #print(self.weight)

            prev_len = len(self.C)

            y_bar_dash, loss_val = self.fun_argmax_y(self.x_train_n, self.y_train_n, self.weight)

            phi = 0  

            for i in range(prev_len):
                temp = self.loss_fun[i] - np.dot(self.weight, psi_original_data - self.psi(self.x_train_n, self.C[i]))
                phi = max(phi,max(0, temp))
               
            condition = loss_val - np.dot(self.weight, psi_original_data - self.psi(self.x_train_n, y_bar_dash))

            if(condition > (phi + self.epsilon)):
                self.C.append(y_bar_dash)
                #print(len(self.C))
                self.loss_fun.append(loss_val)
                self.weight, phi = self.SVM_Multi(self.weight, phi)
                counter_update = 1
                if(len(self.C) % 5 == 0):
                    print(len(self.C))
                    inter_pred = self.Predict(self.weight) 
                    print("Intrermediate F1 score: {}".format(inter_pred))
                    candidate_set.append(len(self.C))
                    F1_score_inter.append(inter_pred)      
        
        #print(len(self.C))        
        #print(self.weight)
        return self.weight, candidate_set , F1_score_inter

    def Main_ROCA(self):

        psi_original_data = self.psi(self.x_train_n, self.y_train_n)
        
        counter_update = 1 
        
        count = 0

        candidate_set = []
        ROCA_inter = []

        while(counter_update and count < 10):
        #while(counter_update):

            counter_update = 0 
            count = count + 1
            
            #print(self.weight)

            prev_len = len(self.C)

            c_dash , loss_val = self.ROCA_argmax(self.x_train_n, self.y_train_n, self.weight)

            y_bar_dash = np.sign(c_dash)
        
            phi = 0  

            for i in range(prev_len):
                temp = self.loss_fun[i] - np.dot(self.weight, psi_original_data - self.psi(self.x_train_n, self.C[i]))
                print("Temp: {}".format(temp))
                print("Loss fun: {}".format(self.loss_fun[i]))
                phi = max(phi,max(0, temp))
               
            condition = loss_val - np.dot(self.weight, psi_original_data - self.psi(self.x_train_n, y_bar_dash))

            print("vector testing: {}".format(psi_original_data - self.psi(self.x_train_n, y_bar_dash)))
            print("loss_val: {}".format(loss_val))
            print("Phi: {}".format(phi))
            print("Condition: {}".format(condition))

            if(condition > (phi + self.epsilon)):
                self.C.append(y_bar_dash)
                #print(len(self.C))
                self.loss_fun.append(loss_val)
                self.weight, phi = self.SVM_Multi(self.weight, phi)
                counter_update = 1
                if(len(self.C) % 1 == 0):
                    print(len(self.C))
                    #inter_pred = self.Predict_ROCA(self.weight) 
                    #print("Intrermediate ROCA loss: {}".format(inter_pred))
                    #loss_val = self.ROCA_loss(Y, c_dash)
                    candidate_set.append(len(self.C))
                    ROCA_inter.append(loss_val)      
        
        #print(len(self.C))        
        #print(self.weight)
        return loss_val, self.weight, candidate_set , ROCA_inter


    def Predict(self, weight_final):
        y_hat = np.dot(self.x_test, weight_final)
        y_hat[y_hat > 0] = 1
        y_hat[y_hat < 0] = -1
        #print("reach in predict function")
        print("Prediction vector y_hat: {}".format(np.unique(y_hat, return_counts= True)))
        precision = metrics.precision_score(self.y_test, y_hat)
        recall = metrics.recall_score(self.y_test, y_hat)
        f1 = self.f1_score_pr(precision, recall)
        #print("f1 score : {}".format(f1))
        return f1

    #def Predict_ROCA(self, weight_final):
    #    y_hat = np.dot(self.x_test, weight_final)
    #    y_hat[y_hat > 0] = 1
    #    y_hat[y_hat < 0] = -1

    #    print("Prediction vector y_hat: {}".format(np.unique(y_hat, return_counts= True)))

        #return loss
    
    
