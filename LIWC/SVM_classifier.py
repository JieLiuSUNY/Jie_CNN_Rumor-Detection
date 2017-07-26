
# Author: Jie Liu
from sklearn import svm
from numpy import genfromtxt
import numpy as np
from sklearn import preprocessing

X = []
y = []

X = genfromtxt('F_trainData.csv', delimiter=',')
X = preprocessing.scale(X)
y = np.array([0] * 51 +[1]*60)

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)
print "Model accuracy = ", clf.score(X,y)


X_test = genfromtxt('F_testData.csv', delimiter=',')
y_pred = clf.predict(X_test)
print y_pred
