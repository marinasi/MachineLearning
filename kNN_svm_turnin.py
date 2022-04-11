import numpy as np
import time
from sklearn import metrics
import skimage.measure
from keras.datasets import fashion_mnist
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("Shape of x_train: {}".format(x_train.shape))
print("Shape of y_train: {}".format(y_train.shape))
print()
print("Shape of x_test: {}".format(x_test.shape))
print("Shape of y_test: {}".format(y_test.shape))

labelNames = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0
print("X_train[0] : \n")
print(x_train[0])
################################################ Resize ############################################################

resize_factor=4

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

print(x_train[0])
#print(x_test)

"""
# KNN Model
start2 = time.time()

knn = KNeighborsClassifier(n_neighbors=10,metric='euclidean')
#knn = KNeighborsClassifier(n_neighbors=10,metric='cosine')

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
print(metrics.classification_report(y_test, y_pred_knn))



#SVM Model
'''start3 = time.time()

svm = SVC(kernel="linear")
svm = SVC(kernel="rbf")   #Gaussian
#svm = SVC(kernel=metrics.pairwise.cosine_similarity)

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
print("Accuracy score: {}".format(svc_accuracy))''' """