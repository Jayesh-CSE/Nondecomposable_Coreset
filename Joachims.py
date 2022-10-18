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

    def __init__(self, x_train, y_train, x_test, y_test, R=1):
        
        self.epsilon = 1
        self.C = []
        self.loss_fun = []
        self.condition_list = []  # edited here
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

    def MCC(self, tp, fp, fn, tn):
        numerator = (tp*tn) - (fp*fn)
        denominator = np.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
        if(denominator == 0):
            denominator = 1
        return numerator/denominator

    def Contingency_Table(self, y_actual, y_hat):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
                TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                FP += 1
            if y_actual[i]==y_hat[i]==-1:
                TN += 1
            if y_hat[i]==-1 and y_actual[i]!=y_hat[i]:
                FN += 1

        return(TP, FP, TN, FN)
    
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

                #loss_delta = self.f1_score_new(a,b,c,d)

                loss_delta = 100*(1-self.f1_score_new(a,b,c,d))
                
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
             
    def objective(self, A):

        W = A[:-1]
        phi_m = A[-1]
        return np.square(LA.norm(W, 2)) * 0.5 +  self.R * phi_m

    def preprocess_constraint(self, A):
    
        W = A[:-1]
        phi_m = A[-1]
        l = len(self.C)
        psi_original_data = self.psi(self.x_train_n, self.y_train_n)
        Mat = []
        B = self.loss_fun

        param = np.append(W, [phi_m])

        for i in range(l):
            Mat.append(psi_original_data - self.psi(self.x_train_n, self.C[i]))
            Mat[i] = np.append(Mat[i], [-1])

        cond = np.dot(Mat, param) - B
        return cond

    def constraint(self, A):
        return self.preprocess_constraint(A)

    def SVM_Multi(self, weight, phi_m):
    
        weight_phi = np.append(weight, [phi_m])
        
        A = [weight_phi]

        constraint_dict = ({'type':'ineq', 'fun': self.constraint})
        
        b_phi = (0, np.inf)
        b_weight = (-np.inf, np.inf)

        demo = [(b_weight)]*(self.x_train_n.shape[1])
        demo.append(b_phi)
        bnds = tuple(demo)

        #sol = minimize(self.objective, A[0] , method = 'SLSQP', bounds=bnds, constraints = constraint_dict, options={'maxiter':500})
        sol = minimize(self.objective, A[0] , method = 'SLSQP', bounds=bnds, constraints = constraint_dict)

        return sol.x[:-1] , sol.x[-1]

    def f1_score_pr(self, precision, recall):
        
        denom = precision+ recall
        
        if(denom == 0):
            denom = 1
        f1 = 2 * (precision * recall) / (denom)

        return f1

    def Main(self):

        psi_original_data = self.psi(self.x_train_n, self.y_train_n)

        #print(psi_original_data)
        
        counter_update = 1 
        
        count = 0

        candidate_set = []
        F1_score_inter = []

        size_c = len(self.x_train_n)
        print("Size of dataset is : {}".format(size_c))

        phi = 0
        
        while(counter_update and (count <= size_c)):

        #while(counter_update and count < 20):
        #while(counter_update):

            counter_update = 0 
            count = count + 1
            
            #print(self.weight)

            prev_len = len(self.C)

            y_bar_dash, loss_val = self.fun_argmax_y(self.x_train_n, self.y_train_n, self.weight)

            
            #if(count == 1):
            #    phi = 0

            # edit code here..... 

            #print("C list is : {}".format(self.C))
            #print("y_bar_dash for current iteration is : {}".format(y_bar_dash))

            #for i in range(prev_len):
            
            if(count != 1):    
                temp = self.loss_fun[prev_len-1] - np.dot(self.weight, psi_original_data - self.psi(self.x_train_n, self.C[prev_len-1])) 

                #temp = self.condition_list[i]

                print("Temp value is : {}".format(temp))
                #print("Weight vector for temp : {}".format(self.weight))

                #print("First part loss_fun value is: {},  Temp value is : {}".format(self.loss_fun[i],temp))
                #print("Second part of minus : {}".format(self.psi(self.x_train_n, self.C[i])))
                #print("i : {} and C[i] : {}".format(i, self.C[i]))

                #print("First part loss_fun value is : {}".format(self.loss_fun[i]))
                #print("Second part is : {}".format(self.psi(self.x_train_n, self.C[i])))
                phi = max(phi,max(0, temp))
                print("Phi value after max : {}".format(phi))

            condition = loss_val - np.dot(self.weight, psi_original_data - self.psi(self.x_train_n, y_bar_dash))

            print("Loss value first part : {}, Condition is: {} and Phi + epsilon is : {}".format(loss_val, condition, (phi + self.epsilon)))
            #print("Weight vector for outer condition : {}".format(self.weight))
            #print("Second part of outer condition minus : {}".format(self.psi(self.x_train_n, y_bar_dash)))

            if(condition > (phi + self.epsilon)):
                self.C.append(y_bar_dash)
                #print(len(self.C))
                self.loss_fun.append(loss_val)
                self.condition_list.append(condition)  # wont work this as we are updating the weight and that is making our temp very small....
                self.weight, psi_dummy = self.SVM_Multi(self.weight, phi) #why phi value is changing here as we wanted to maximize this value which is most violated constraint so far, so why do we minimize it with svm ...??? edit code here.....

                print("SVM calculated psi is: {}".format(psi_dummy))
                
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

    def Predict_mcc(self, weight_final):
        y_hat = np.dot(self.x_test, weight_final)
        y_hat[y_hat > 0] = 1
        y_hat[y_hat < 0] = -1
        
        print("Prediction vector y_hat: {}".format(np.unique(y_hat, return_counts= True)))
        
        T = Contingency_Table(self.y_test, y_hat)
        
        MCC_score = MCC(T[0], T[1], T[3], T[2])
        
        return MCC_score
