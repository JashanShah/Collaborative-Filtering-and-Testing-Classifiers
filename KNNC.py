import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.datasets import fetch_openml
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
#Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

# rescale the data, use the traditional train/test split
# (60K: Train) and (10K: Test)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]



#1
KNNC = KNeighborsClassifier(n_neighbors=6)
KNNC.fit(X_train, y_train)
predict = KNNC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)

#2
KNNC = KNeighborsClassifier(n_neighbors=7, weights='distance')
KNNC.fit(X_train, y_train)
predict = KNNC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)

#3
KNNC = KNeighborsClassifier(n_neighbors=6, weights='distance')
KNNC.fit(X_train, y_train)
predict = KNNC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)

#4
KNNC = KNeighborsClassifier(n_neighbors=3,algorithm="kd_tree",leaf_size=30,weights='distance')
KNNC.fit(X_train, y_train)
predict = KNNC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)


#5
KNNC = KNeighborsClassifier(n_neighbors=5,algorithm="kd_tree",leaf_size=30, weights='distance')
KNNC.fit(X_train, y_train)
predict = KNNC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)


#6
KNNC = KNeighborsClassifier(algorithm='brute', n_jobs=5)
KNNC.fit(X_train, y_train)
predict = KNNC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)


#7
KNNC = KNeighborsClassifier(algorithm='brute', n_jobs=10, weights='distance')
KNNC.fit(X_train, y_train)
predict = KNNC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)



#8
KNNC = KNeighborsClassifier(n_neighbors=10,algorithm='brute', n_jobs=10, weights='distance')
KNNC.fit(X_train, y_train)
predict = KNNC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)


#9
KNNC = KNeighborsClassifier(n_neighbors=15,algorithm='brute', n_jobs=10, weights='distance', leaf_size=50)
KNNC.fit(X_train, y_train)
predict = KNNC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)



#10
KNNC = KNeighborsClassifier(n_neighbors=2, weights='distance', algorithm='ball_tree', n_jobs=3)
KNNC.fit(X_train, y_train)
predict = KNNC.predict(X_test)
accScore = accuracy_score(y_test, predict)
print(accScore)