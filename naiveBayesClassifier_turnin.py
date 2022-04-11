import collections
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import time
from sklearn import metrics
from IPython.display import Markdown, display
import random
import matplotlib.pyplot as plt
import skimage.measure
import tensorflow as tf
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("Shape of x_train: {}".format(x_train.shape))
print("Shape of y_train: {}".format(y_train.shape))
print()
print("Shape of x_test: {}".format(x_test.shape))
print("Shape of y_test: {}".format(y_test.shape))

labelNames = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

# Normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0


###########################################################################################################################

############################### Create Dictionaries and calculate Mean, Stantard Deviation#################################

vectorsByClass = {0:list(),1:list(),2:list(),3:list(),4:list(),5:list(),6:list(),7:list(),8:list(),9:list()}

for i in range(len(x_train)):
    vectorsByClass[y_train[i]].append(x_train[i])

statisticsForStd={0:list(),1:list(),2:list(),3:list(),4:list(),5:list(),6:list(),7:list(),8:list(),9:list()}
statisticsForMean={0:list(),1:list(),2:list(),3:list(),4:list(),5:list(),6:list(),7:list(),8:list(),9:list()}

        
for category in vectorsByClass:
    zipped = zip(*vectorsByClass[category])
    listOfElementsByColumn = list(zipped)
    print(len(listOfElementsByColumn))
    for element in range(len(listOfElementsByColumn)):
        statisticsForMean[category].append(np.average(listOfElementsByColumn[element]))
        if(np.std(listOfElementsByColumn[element])!=0):
            
        
            statisticsForStd[category].append(np.std(listOfElementsByColumn[element]))
        else:
            
        
            statisticsForStd[category].append(0.0000000000000001)

#########################################################################################################################
            
#################################### Calculate Probs and Test Results ###################################################

test_results = []

for testVectorIndex in range(len(x_test)):

    g = 0
    predicted_category = 0
    
    for categoryIndex in range(10):
        
        totalSum1 = 0
        totalSum2 = 0

        for featureIndexOfCategory in range(len(statisticsForStd[categoryIndex])):
            totalSum1 = totalSum1 + np.log(statisticsForStd[categoryIndex][featureIndexOfCategory])

        for featureIndex in range(len(x_test[testVectorIndex])):
            totalSum2 = totalSum2 + (((x_test[testVectorIndex][featureIndex] - statisticsForMean[categoryIndex][featureIndex])**2)/(2*(statisticsForStd[categoryIndex][featureIndex]**2)))

        tempResult = (-1)*totalSum1-totalSum2

        if g < tempResult:
            g = tempResult
            predicted_category = categoryIndex


    test_results.append(predicted_category)

correct = 0

for true_label_Index in range(len(y_test)):
    if y_test[true_label_Index] == test_results[true_label_Index]:
        correct = correct +1
knn_f1 = metrics.f1_score(y_test,test_results, average= "weighted")

print("Correct : " + str(correct) + "\n")
print("Accuracy : " + str(correct/len(y_test)) + "\n")
print("F1 score: {}".format(knn_f1))





