import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.datasets import fetch_openml
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
#MLPClassifier(hidden_layer_sizes=(100,), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
#Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

# rescale the data, use the traditional train/test split
# (60K: Train) and (10K: Test)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


#1
MLPC = MLPClassifier(solver='sgd', activation='logistic',alpha=1,learning_rate_init=.1,early_stopping=True)
MLPC.fit(X_train, y_train)
predict = MLPC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)

#2
MLPC = MLPClassifier(activation = 'identity')
MLPC.fit(X_train, y_train)
predict = MLPC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)

#3
MLPC = MLPClassifier(activation = 'logistic')
MLPC.fit(X_train, y_train)
predict = MLPC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)

#4
MLPC = MLPClassifier(activation='relu', solver='adam',hidden_layer_sizes=(15,),random_state=1)
MLPC.fit(X_train, y_train)
predict = MLPC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)

#5
MLPC = MLPClassifier(solver='sgd', random_state=1, batch_size ='auto')
MLPC.fit(X_train, y_train)
predict = MLPC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)

#6
MLPC = MLPClassifier(solver='adam', validation_fraction=0.2, beta_1=0.9, beta_2=0.999)
MLPC.fit(X_train, y_train)
predict = MLPC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)

#7
MLPC = MLPClassifier(solver='sgd', nesterovs_momentum = False)
MLPC.fit(X_train, y_train)
predict = MLPC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)

#8
MLPC = MLPClassifier(solver='sgd', random_state=2, batch_size ='auto')
MLPC.fit(X_train, y_train)
predict = MLPC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)


#9
MLPC = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=500)
MLPC.fit(X_train, y_train)
predict = MLPC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)


#10
MLPC = MLPClassifier(activation='relu',solver='lbfgs',  hidden_layer_sizes=(10,10),random_state=35, alpha=0.001,learning_rate='constant',learning_rate_init = 0.7,shuffle=True, max_iter = 100)
MLPC.fit(X_train, y_train)
predict = MLPC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)