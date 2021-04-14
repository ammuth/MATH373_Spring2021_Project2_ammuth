# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 19:20:29 2021

@author: lngsm
"""
#https://www.youtube.com/watch?v=e90IB67Q0q8
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(7)



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

def logReg_SGD(X, Y, alpha, batch):
    numEpochs = 5
    N, d = X.shape
    X = np.insert(X, 0, 1, axis = 1)
    K = Y.shape[1]
    batch_size = batch
    batch_idx = 0
    
    beta = np.zeros((K, d+1))
    Lvals = []
    
    for ep in range(numEpochs):
        
        L = eval_L(X, Y, beta)
        Lvals.append(L)
        
        print("Epoch is: " + str(ep) + " Cost is: " + str(L))
        
        #prm = np.random.permutation(N)

        #for i in prm:
        for i in range(0, N, batch_size):
            stop_idx = i + batch_size
            stop_idx = min(stop_idx, N)
            num_examples_in_batch = stop_idx - i
            
            XiHat = X[i]
            Yi = Y[i]
            
            qi = softmax(beta @ XiHat)
            grad_Li = np.outer(qi-Yi, XiHat)
            
            beta = beta - alpha*grad_Li
        grad_Li = grad_Li/num_examples_in_batch
    return beta, Lvals
            
def predictLabels(X, beta):
    X = np.insert(X, 0, 1, axis =1)
    N = X.shape[0]
    predictions = []
    probabilities = []
    
    for i in range(N):
        XiHat = X[i]
        qi = softmax(beta @ XiHat)
        
        k = np.argmax(qi)
        predictions.append(k)
        probabilities.append(np.max(qi))
        
    return predictions, probabilities
        
        

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train/255.0
X_test = X_test/255.0

N_train, numRows, numCols = X_train.shape
X_train = np.reshape(X_train, (N_train, numRows*numCols))

Y_train = pd.get_dummies(Y_train).values

alpha = 0.01

beta, Lvals = logReg_SGD(X_train, Y_train, alpha,4)


plt.semilogy(Lvals)
plt.show()
#%%
N_test = X_test.shape[0]
#%%
X_test = np.reshape(X_test, (N_test, numRows*numCols))
#%%
predictions, probabilities = predictLabels(X_test, beta)

#%%
probabilities = np.array(probabilities)
agreement = (predictions == Y_test)
sortedIdxs = np.argsort(probabilities)
sortedIdxs = sortedIdxs[::-1]

difficultExamples = []

for i in sortedIdxs:
    if agreement[i] == False:
        difficultExamples.append(i)
        
   
#%%
numCorrect = 0
for i in range(N_test):
    if predictions[i] == Y_test[i]:
        numCorrect += 1

accuracy = numCorrect/N_test
print('Accuracy: ' + str(accuracy))
#%%
##run this sperately from the rest of code
probabilities[difficultExamples[0:5]]

i = difficultExamples[0]
Xi = np.reshape(X_test[i], (28,28))
plt.imshow(Xi) 
plt.show()
print("Difficult example #1 predicted value: " + str(predictions[i]))
print("Difficult example #1 actual value: " + str(Y_test[i]))

i = difficultExamples[1]
Xi = np.reshape(X_test[i], (28,28))
plt.imshow(Xi) 
plt.show()
print("Difficult example #1 predicted value: " + str(predictions[i]))
print("Difficult example #1 actual value: " + str(Y_test[i]))

i = difficultExamples[2]
Xi = np.reshape(X_test[i], (28,28))
plt.imshow(Xi) 
plt.show()
print("Difficult example #1 predicted value: " + str(predictions[i]))
print("Difficult example #1 actual value: " + str(Y_test[i]))

