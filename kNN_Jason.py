
from scipy.io import arff
import pandas as pd
import numpy as np 
import sys
import math
import time 

start = time.time()

file_name = (sys.argv[1])
k = int(sys.argv[2])
mode = (sys.argv[3])

def euclidian(dp_one, dp_two):
    dis = 0
    for a in range(len(dp_one)):
        dis += (dp_one[a] - dp_two[a])**2
    return dis

def manhattan(dp_one, dp_two):
    dis = 0 
    for a in range(len(dp_one)):
         dis += abs((dp_one[a] - dp_two[a]))
    return dis


def minkowski(dp_one, dp_two):
    dis = 0
    for a in range(len(dp_one)):
        dis += (abs(dp_one[a] - dp_two[a]))**3
    return dis


def majority(class_list):
    return max(set(class_list), key = class_list.count)
 

def kNN(file_location, k, mode):
    data = arff.loadarff(file_location)
    df = pd.DataFrame(data[0])

    target = df['class']
    df = df.drop(['class'], axis = 1)

    Data = df.values
    Target = target.values

    predictions =  np.zeros(shape = (len(Target), 2))
    indexX = 0
    for X in Data:
        top_dis = []
        top_dis_class = []
        indexY = 0
        for Y in Data: 
            dis = 0
            if mode == "euclidian":
                dis = euclidian(X,Y)
            if mode == "manhattan":
               dis = manhattan(X,Y)
            if mode == "minkowski":
                dis = minkowski(X,Y)
            if indexX == indexY:
                dis = np.inf
            if len(top_dis) < k:
                top_dis_class.append(Target[indexY])
                top_dis.append(dis)
            elif dis < max(top_dis):
                top_dis_class.pop(top_dis.index(max(top_dis)))
                top_dis_class.append(Target[indexY])
                top_dis.remove(max(top_dis))
                top_dis.append(dis)
            indexY += 1
        mostCommon = majority(top_dis_class)
        predictions[indexX] = [indexX, mostCommon]
        indexX += 1
    return predictions, Target

def accuracy(predictions, Target):
    correct = 0
    for x in range(len(Target)):
        if Target[x] == predictions[x,1]:
            correct += 1
    accuracy = correct / len(Target)
    return accuracy


predictions, Target = kNN(file_name, k, mode)
print("Accuracy was:", accuracy(predictions, Target))

end = time.time()
total_time = end-start
print("Execution time: ", total_time, " seconds")