# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#Functions for activations:
def relu(x):
    y = x * (x > 0)
    return y
def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y
def reluDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
#Vectorization
X_train_nn = X_train.T
y_train_nn = y_train.as_matrix()
m = y_train_nn.shape[0]
layer_dims = [8, 1] 

#Initializing parameteres
W1 = np.random.randn(layer_dims[0],X_train_nn.shape[0])
W2 = np.random.randn(layer_dims[1], layer_dims[0])
b1 = np.zeros(shape = (layer_dims[0],1))
b2 = np.zeros(shape = (layer_dims[1],1))
from sklearn.metrics import f1_score
param_cache = {"W1":[],"W2":[], "b1": [], "b2": [], "cost": [],
               "learning_rate":[],
               "f1 score":[]}
learning_rate = 0.00001
while True:
    for i in range(1000):
        #Checking number of parameters in the list, if bigger than 150k then drop
        if len(param_cache['cost'])>=150000:
            param_cache = {"W1":[],"W2":[], "b1": [], "b2": [], "cost": [],
               "learning_rate":[],
               "f1 score":[]}
        #Forward propagation:
        Z1 = np.dot(W1,X_train_nn) + b1
        A1 = relu(Z1)
        Z2 = np.dot(W2,A1) + b2
        A2 = sigmoid(Z2)
        # Calculation cost
        cost = -(1/m)*np.sum(y_train_nn.T*np.log(A2)+(1-y_train_nn.T)*np.log(1-A2))
        param_cache['W1'].append(W1)
        param_cache["W2"].append(W2)
        param_cache["b1"].append(b1)
        param_cache["b2"].append(b2)
        param_cache["cost"].append(cost)
        y_pred = (A2>0.5)*1
        y_pred = y_pred.T
        param_cache['f1 score'].append(f1_score(y_train_nn, y_pred))
        #Backward propagation:
        dZ2 = A2 - y_train_nn
        dW2 = (1/A1.shape[1])*np.dot(dZ2, A1.T)
        db2 = (1/A1.shape[1])*np.sum(dZ2, axis = 1, keepdims = True)
        dZ1 = np.dot(W2.T, dZ2)*reluDerivative(Z1)#(1- np.tanh(Z1)**2)
        dW1 = (1/X_train_nn.shape[1])*np.dot(dZ1, X_train_nn.T)
        db1 = (1/X_train_nn.shape[1])*np.sum(dZ1, axis = 1, keepdims = True)
        W1 = W1 - learning_rate*dW1
        W2 = W2 - learning_rate*dW2
        b1 = b1 - learning_rate*db1
        b2 = b2 - learning_rate*db2

    print(cost)
    
    if (param_cache['cost'][-2] - param_cache['cost'][-1])<0:
        break
print(param_cache['cost'].index(min(param_cache['cost'])))
print(param_cache['f1 score'][param_cache['cost'].index(min(param_cache['cost']))])
from sklearn.metrics import accuracy_score
accuracy_score(y_train_nn, y_pred)
import csv
with open('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\parameters_1hidden.csv', 'w') as f:
    w = csv.DictWriter(f, param_cache.keys())
    w.writeheader()
    w.writerow(param_cache)
#test
X_test_nn = X_test.T
W1_test = param_cache["W1"][56998]
W2_test = param_cache["W2"][56998]
b1_test = param_cache["b1"][56998]
b2_test = param_cache["b2"][56998]
Z1_test = np.dot(W1_test,X_test_nn) + b1_test
A1_test = relu(Z1_test)
Z2_test = np.dot(W2_test,A1_test) + b2_test
A2_test = sigmoid(Z2_test)
y_pred_t = (A2_test>0.5)*1
y_pred_t = y_pred_t.T   
np.save('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\W1_1hl_relu_6l_16f', W1)
np.save('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\W2_1hl_relu_6l_16f', W2)
np.save('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\b1_1hl_relu_6l_16f', b1)
np.save('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\b2_1hl_relu_6l_16f', b2)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred_t[:, 0]
    })
submission.to_csv('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\submission_nn_13.csv', 
                  index = False)
W1 = np.load('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\W1_1hl.npy')
W2 = np.load('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\W2_1hl.npy')
b1 = np.load('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\b1_1hl.npy')
b2 = np.load('C:\\Users\\maksim.lebedev\\Desktop\\MachineLearning\\Kaggle\\Titanic\\b2_1hl.npy')