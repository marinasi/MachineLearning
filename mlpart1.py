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


# Helper function
# Print markdown style
def printmd(string):
    display(Markdown(string))


# cm is the confusion matrix, names are the names of the classes.
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=90)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("Shape of x_train: {}".format(x_train.shape))
print("Shape of y_train: {}".format(y_train.shape))
print()
print("Shape of x_test: {}".format(x_test.shape))
print("Shape of y_test: {}".format(y_test.shape))

labelNames = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
sample = 1902
each = x_train[sample]

plt.figure(figsize=(3,3))
plt.imshow(each)
plt.colorbar()
plt.show()
print("Image (#{}): Which is label number '{}', or label '{}''".format(sample,y_train[sample], labelNames[y_train[sample]]))

ROW = 7
COLUMN = 7
plt.figure(figsize=(10, 10))
for i in range(ROW * COLUMN):
    temp = random.randint(0, len(x_train) + 1)
    image = x_train[temp]
    plt.subplot(ROW, COLUMN, i + 1)
    plt.imshow(image, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(labelNames[y_train[temp]])
    plt.tight_layout()

plt.show()

# Normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

################################################ Resize ############################################################

resize_factor=2

x_train_resized=[]
x_test_resized=[]

for i in range(0,len(x_train)):
    x_train_resized.append(skimage.measure.block_reduce(x_train[i], (resize_factor,resize_factor), np.average))

for i in range(0,len(x_test)):
    x_test_resized.append(skimage.measure.block_reduce(x_test[i], (resize_factor,resize_factor), np.average))

x_train = np.asarray(x_train_resized)
x_test = np.asarray(x_test_resized)

####################################################################################################################

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print(x_train.shape)
print(x_test.shape)


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



# KNN Model
'''start2 = time.time()

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)

end2 = time.time()
knn_time = end2-start2

print("KNN Time: {:0.2f} minute".format(knn_time/60.0))

knn_f1 = metrics.f1_score(y_test, y_pred_knn, average= "weighted")
knn_accuracy = metrics.accuracy_score(y_test, y_pred_knn)
knn_cm = metrics.confusion_matrix(y_test, y_pred_knn)
print("-----------------K-nearest neighbors Report---------------")
print("F1 score: {}".format(knn_f1))
print("Accuracy score: {}".format(knn_accuracy))
print("Confusion matrix: \n", knn_cm)
print('Plotting confusion matrix')

plt.figure()
plot_confusion_matrix(knn_cm, labelNames)
plt.show()

print(metrics.classification_report(y_test, y_pred_knn))'''

'''start3 = time.time()

svm = SVC(kernel="linear")
svm.fit(x_train, y_train)
y_pred_svc = svm.predict(x_test)

end3 = time.time()
svm_time = end3-start3
svc_f1 = metrics.f1_score(y_test, y_pred_svc, average= "weighted")
svc_accuracy = metrics.accuracy_score(y_test, y_pred_svc)
svc_cm = metrics.confusion_matrix(y_test, y_pred_svc)
print("SVM Time: {:0.2f} minute".format(svm_time/60.0))
print("-----------------SVM Report---------------")
print("F1 score: {}".format(svc_f1))
print("Accuracy score: {}".format(svc_accuracy))
print("Confusion matrix: \n", svc_cm)
print('Plotting confusion matrix')'''

'''from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(500,200),activation='logistic',random_state=1)'''

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(len(x_train[0]),)),
    #tf.keras.layers.Dense(500, activation='sigmoid'),
    tf.keras.layers.Dense(200, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')
])

opt = tf.keras.optimizers.SGD(learning_rate=0.1)

model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

model.fit(x_train, y_train, epochs=10)
test_acc = model.evaluate(x_test,  y_test)
print('\nTest accuracy:', test_acc)