# Create first network with Keras
# http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 12
numpy.random.seed(seed)
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


# create model
model = Sequential()
model.add(Dense(256, input_dim=13, init='uniform', activation='relu'))
model.add(Dense(128, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(XTest1, YTest1, nb_epoch=500, batch_size=90)
# evaluate the model
print("\n\n\n Evaluation")
scores = model.evaluate(XTest1, YTest1)
print("\n %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("%s: %.2f" % (model.metrics_names[0], scores[0]))

print("\n\n\n Validation")
scores = model.evaluate(XTest1Valid, YTest1Valid)
print("\n %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("%s: %.2f" % (model.metrics_names[0], scores[0]))

# calculate predictions
predictions = model.predict(XTest)
# round predictions
rounded = [(float)(numpy.round_(x)) for x in predictions]
#print(rounded)
