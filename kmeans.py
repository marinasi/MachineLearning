import numpy
import random
import math
from keras.datasets import fashion_mnist

########################################################################

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("Shape of x_train: {}".format(x_train.shape))
print("Shape of y_train: {}".format(y_train.shape))
print()
print("Shape of x_test: {}".format(x_test.shape))
print("Shape of y_test: {}".format(y_test.shape))

x_train_flatten=[]
for i in range(0,len(x_train)):
    x_train_flatten.append(x_train[i].flatten())


x_train_flatten_np = numpy.array(x_train_flatten)    
# Normalize the data
x_train_flatten_np  =x_train_flatten_np.astype('float32')

x_train_flatten_np  /= 255.0

print(x_train_flatten_np [0])
print(x_train_flatten_np .shape)
print(len(x_train_flatten_np[0]))

x_train_flatten=x_train_flatten_np.tolist()
#########################################################################

M = 10

Centers = []
Clusters = []

def initialization():
    
    for i in range(0,M):
        rC = random.randint(0, len(x_train_flatten))
        #center=[]
        feature=x_train_flatten[rC]
        #center.append(feature)
        Centers.append(feature)
        newCluster = []
        Clusters.append(newCluster)

def calculateClusterMean(cluster):
    sums=[0] * len(x_train_flatten[0])
    for element in cluster:
        for i in range(0,len(element)):
            sums[i]=sums[i] +element[i]
    #newCenter = sums/len(cluster)
    newCenter = [x/len(cluster) for x in sums]
    return newCenter

def newCenterCalculation():
    for i in range(0,len(Clusters)):
        Centers[i] = calculateClusterMean(Clusters[i])
        
def removeFromCluster(vectorToRemove):
    for cluster in Clusters:
        for vector in cluster:
            if vector == vectorToRemove:
                cluster.remove(vectorToRemove)
        
def performKmeans():
    
    initialization() # arxikopoiisi kentrwn
    
    terminationFlag = False
    
    while terminationFlag==False:
        
        changeFlag = False
        
        for i in range(0,len(x_train_flatten)): # kathe eikona sto set ekpaideusis
            
            minD = 100000000000000
            index =0
            
            print(i)
            
            for j in range(0,len(Centers)): # kathe kentro
                
                current = numpy.linalg.norm(x_train_flatten_np[i] - Centers[j]) # ypologismos Eukleidias apostasis 
                
                if current < minD:
                    
                    minD = current
                    index = j
            
            
            if x_train_flatten[i] not in Clusters[index]:
                removeFromCluster(x_train_flatten[i]) # afairesi apo to cluster pou vrisketai i eikona
                Clusters[index].append(x_train_flatten[i]) # topothetisi sto kainourgiio cluster
                print(len(Clusters[index]))
                changeFlag=True
                
        newCenterCalculation() # ypologismos tou neou kentrou tou cluster
        
        if(changeFlag==False):
            terminationFlag=True
            
        for q in range(0,M):
            print("Cluster "+str(q)+" : "+str(len(Clusters[q])))
        print("-------------------")
     
performKmeans()
for i in range(0,M):
    print(len(Clusters[i]))