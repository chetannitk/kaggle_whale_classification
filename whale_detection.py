# import the necessary packages
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
import cv2
import sys
import csv
import os.path
from numpy import genfromtxt
#Local modules
from train_map import *
import cPickle as pickle

## Grab the  dataset (if this is the first time you are running (i.e train and 
## test pickle files are not there in current directory.)
## Serialize and Deserialize the CSV reader and store the train and test
## object in pickle files.
if not (os.path.isfile("train.pkl") and os.path.isfile("test.pkl")):
    my_train_data = genfromtxt('train_whale.csv', delimiter=',')
    my_test       = genfromtxt('test_whale.csv', delimiter=',')
    train_in = open("train.pkl", "wb")
    test_in = open("test.pkl", "wb")
    pickle.dump(my_train_data, train_in, pickle.HIGHEST_PROTOCOL)
    pickle.dump(my_test, test_in , pickle.HIGHEST_PROTOCOL)
    train_in.close()
    test_in.close()
else:
    train_in = open("train.pkl", "rb")
    test_in = open("test.pkl", "rb")
    my_train_data = pickle.load(train_in)
    my_test = pickle.load(test_in)

## Store the test image name in testY.
with open('test_whale.csv') as test_h:
    test_matrix  = csv.reader(test_h, delimiter=",")
    testY = []
    for row in test_matrix:
        testY.append(row[-1])
my_test_data = my_test[:,:-1]
print "[X] Training Starting..."
dataset = np.array(my_train_data, np.int32)
test_dataset = np.array(my_test_data, np.int32)
# scale the data to the range [0, 1] 
normalized_train = dataset[:,0:-1]/255.0
trainX = normalized_train[:,:6401]
trainY = dataset[:,-1]
#print len(trainY)
testX = test_dataset[:,:]/255.0
print testX
print trainX.shape
print "trainX type "+str(type(trainX))
print "trainY type "+str(type(trainY))
print "testX type  "+str(type(testX))
# train the Deep Belief Network with 6400 input units (the flattened,
# 80x80 resized grayscale image), 300 hidden units, dynamic output units (one for
# each possible output classification.)
dbn = DBN(
    [trainX.shape[1], 500, len(np.unique(trainY))],
    learn_rates = 0.2,
    learn_rate_decays = 0.9,
    epochs = 50,
    verbose = 1,)
print trainX.shape
print trainY.shape
dbn.fit(trainX, trainY)

# compute the predictions for the test data and show a classification
# report
preds = dbn.predict(testX)
print np.unique(preds)
for i in range(len(preds)):
    print str(testY[i]),
    print "----whale_"+str(preds[i])

