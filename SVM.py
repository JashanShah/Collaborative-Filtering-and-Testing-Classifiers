import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.datasets import fetch_openml
import time

#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
#Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

# rescale the data, use the traditional train/test split
# (60K: Train) and (10K: Test)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

#1
linearSVMC = svm.SVC(kernel='linear', C=2.0)
linearSVMC.fit(X_train, y_train)
predict = linearSVMC.predict(X_test)
accScore = accuracy_score(y_test, predict) * 100
print("Linear SVMC:")
print(accScore)

#2
linearSVMC = svm.SVC(kernel='linear', C=1.0)
linearSVMC.fit(X_train, y_train)
predict = linearSVMC.predict(X_test)
accScore = accuracy_score(y_test, predict) * 100
print("Linear SVMC:")
print(accScore)

#3
polySVMC = svm.SVC(kernel='poly', gamma='auto', max_iter=5, degree=2)
polySVMC.fit(X_train, y_train)
predict = polySVMC.predict(X_test)
accScore = accuracy_score(y_test, predict) * 100
print("Poly SVMC:")
print(accScore)

#4
SVMC = svm.SVC(kernel='rbf', C=1,decision_function_shape='ovo')
SVMC.fit(X_train, y_train)
predict = SVMC.predict(X_test)
accScore = accuracy_score(y_test, predict) * 100
print(accScore)

#5
SVMC = svm.SVC(kernel='rbf', C=1, max_iter=100)
SVMC.fit(X_train, y_train)
predict = SVMC.predict(X_test)
accScore = accuracy_score(y_test, predict) * 100
print(accScore)

#6
SVMC = svm.SVC(C=0.9, kernel='poly', degree=3)
SVMC.fit(X_train, y_train)
predict = SVMC.predict(X_test)
accScore = accuracy_score(y_test, predict) * 100
print(accScore)

#7
SVMC = svm.SVC(kernel='sigmoid', C=1, class_weight='balanced')
SVMC.fit(X_train, y_train)
predict = SVMC.predict(X_test)
accScore = accuracy_score(y_test, predict) * 100
print(accScore)

#8
SVMC = svm.SVC(C=100, kernel='linear', gamma='auto', max_iter=1000, probability=False, random_state=None)
SVMC.fit(X_train, y_train)
predict = SVMC.predict(X_test)
accScore = accuracy_score(y_test, predict) * 100
print(accScore)

#9
SVMC = svm.SVC(kernel='rbf', gamma='scale', random_state=None)
SVMC.fit(X_train, y_train)
predict = SVMC.predict(X_test)
accScore = accuracy_score(y_test, predict) * 100
print(accScore)

#10
SVMC = svm.SVC(kernel='sigmoid', coef0=0.68,degree=15, random_state=1)
SVMC.fit(X_train, y_train)
predict = SVMC.predict(X_test)
accScore = accuracy_score(y_test, predict) * 100
print(accScore)