
import numpy as np


#def ROCA_argmax(self, X, Y, W):  for incorporating it into class file.

#still needs to figure out how to incorporate this in our pipeline as this is returning vector C and not y-bar-prime

def ROCA_argmax(X, Y, W):   

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
			
	return C
		

def ROCA_loss(Y, C_Prime):

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



	
	
	
	 
