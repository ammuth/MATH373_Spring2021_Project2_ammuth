# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 11:40:20 2021

@author: lngsm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import seaborn
from sklearn.datasets import load_iris
import sklearn as sk
import sklearn.model_selection

#rom sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data[:-1,:], iris.target[:-1]
X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, y)


def softmax(u):
    expu = np.exp(u)
    return expu/np.sum(expu)

def crossEntropy(p, q):
    return -np.vdot(p, np.log(q))

def eval_L(X, Y, beta):
    #Assume feature vectors are augmented already.
    
    N = X.shape[0]
    L = 0.0
    
    for i in range(N):
        XiHat = X[i]
        Yi = Y[i]
        qi = softmax(beta @ XiHat)
        
        L += crossEntropy(Yi, qi)
        
    return L




def gradient_descent(X, Y, alpha):
    #batch_size = batch
    numEpochs = 100
    N, d = X.shape
    X = np.insert(X, 0, 1, axis = 1)
    K = Y.shape[1]
    
    beta = np.zeros((K, d+1))
    Lvals = []
    
    for ep in range(numEpochs):
        
        L = eval_L(X, Y, beta)
        Lvals.append(L)
        
        print("Epoch is: " + str(ep) + " Cost is: " + str(L))
        
        #prm = np.random.permutation(N)
        #prm = prm[0:batch_size]
        #batch_idx = 0
        for i in range(N):
            #stop_idx = i + batch_size
            #stop_idx = min(stop_idx, N)
            #num_examples_in_batch = stop_idx - i
            
            XiHat = X[i]
            Yi = Y[i]
            
            qi = softmax(beta @ XiHat)
            grad_Li = np.outer(qi-Yi, XiHat)
        
            beta = beta - alpha*grad_Li
            
        grad_Li = grad_Li/N

        #batch_idx += 1
            
    return beta, Lvals


#N_train, numRows, numCols = X_train.shape
#X_train = np.reshape(X_train, (N_train, numRows*numCols))


Y_train = pd.get_dummies(Y_train).values

alpha = 0.02


beta, Lvals = gradient_descent(X_train, Y_train, alpha)


plt.semilogy(Lvals)









  
