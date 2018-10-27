import numpy
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier


# load dataset
dataset = numpy.loadtxt("heart.txt", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:13]
Y = dataset[:,13]
XTest1 = dataset[0:91, 0:13]
YTest1 = dataset[0:91, 13]
XTest1 = numpy.concatenate([XTest1, dataset[181:271, 0:13]])
YTest1 = numpy.concatenate([YTest1, dataset[181:271, 13]])
XTest1Valid = dataset[91:181, 0:13]
YTest1Valid = dataset[91:181, 13]

XPredict = dataset[250:271, 0:13]
YPredict = dataset[250:271, 13]

ovr = OneVsRestClassifier(svm.SVC(kernel='linear', C=2))
ovr.fit(X, Y)
cross = cross_val_score(ovr, X, Y, cv=3)
print("Cross-Validation")
print(cross)
scores = ovr.score(XTest1, YTest1)
print("Evalutation: %0.2f%%" % (scores.mean()*100))
scores = ovr.score(XTest1Valid, YTest1Valid)
print("Validation: %0.2f%%" % (scores.mean()*100))

print("Vraies valeurs = ")
print(YPredict)
print("Prédictions = ")
print(ovr.predict(XPredict))


print(ovr.decision_function(XPredict))
