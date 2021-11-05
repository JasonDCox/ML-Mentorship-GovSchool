from scipy.io import arff
import pandas as pd
import numpy as np 
import sys
import time 
import os
import psutil
import matplotlib.pyplot as plt



start = time.time()

file_name = (sys.argv[1])
k = int(sys.argv[2])



# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
  
# decorator function
def profile(kNN):
    def wrapper(*args, **kwargs):
  
        mem_before = process_memory()
        result = kNN(*args, **kwargs)
        mem_after = process_memory()
        print("{}:consumed memory: {:,}".format(
            kNN.__name__,
            mem_after - mem_before))
        global mem_used
        mem_used = mem_after - mem_before
        
  
        return result
    return wrapper
  



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
 
@profile

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
            elif mode == "manhattan":
               dis = manhattan(X,Y)
            elif mode == "minkowski":
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

file = open("ouput.md", "w")



p1, Target = kNN(file_name, k, 'euclidian')
p2, Target = kNN(file_name, k, 'manhattan')
p3, Target = kNN(file_name, k, "minkowski")


predictions =  np.zeros(shape = (len(Target), 2))
for x in range(len(p1)):
    answers = []
    answers.append(p1[x,1])
    answers.append(p2[x,1])
    answers.append(p3[x,1])
    mostCommon = majority(answers)
    predictions[x] = [x, mostCommon]

print(predictions)
print("Accuracy was:", accuracy(predictions, Target))

end = time.time()

file.write("## General Metrics: \n")
file.write("- Accuracy was: " + str(accuracy(predictions, Target)) + "\n" )
total_time = end-start
print("Execution time: ", total_time, " seconds")

file.write("- Execution time: " + str(total_time) + " seconds\n")
file.write("- Consumed Memory: " + str(mem_used) + "\n")

fig, ax = plt.subplots()

predicted_class = [" "]
for x in set(Target):
    predicted_class.append("Predicted: " + str(x))

table_data = [predicted_class]
for row in set(Target):
    line = ["Actual: " + str(row)]
    for column in set(Target):
        count = 0
        for a in predictions:
            if a[1] == column and Target[int(a[0])] == row:
                count += 1
        line.append(count)
    table_data.append(line)
    
table = ax.table(cellText = table_data, loc = 'center')
table.set_fontsize(10)
table.scale(1.25,1.25)
ax.axis('off')

print("-------------------------------")


for x in range(len(set(Target))):
    i = int(x) + 1
    true_negative = len(Target)
    true_positive = table_data[i][i]
    false_negative = 0
    for y in table_data[i][1:]:
        false_negative += y
    true_negative -= false_negative
    false_negative -= true_positive 
    false_positive = 0
    for z in table_data[1:]:
        false_positive += z[i]
    false_positive -= true_positive
    true_negative -= false_positive
    
    
    
    try:
        precision = format((true_positive) / (true_positive + false_positive), "0.3f")
    except:
        precision = "N/A"
    try:
        recall = format((true_positive) / (true_positive + false_negative), "0.3f")
    except:
        recall = "N/A"
    try:
        F1Score = format(2 * (1 / ((1 / float(precision)) + (1 / float(recall)))), "0.3f")
    except:
        F1Score = "N/A"
    try:
        sensitivity = format((true_positive) / (false_negative + true_positive), "0.3f")
    except: 
        sensitivity = "N/A"
    try:
        specificity = format((true_negative) / (true_negative + false_positive), "0.3f")
    except: 
        specificity = "N/A"
        
    print("Metrics for " + str(x) + ":")
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", F1Score)
    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    print("-------------------------------")
    file.write("## Metrics for " + str(x) + ":\n")
    file.write("| Metric | Result |\n")
    file.write("| --- | --- | \n")
    file.write("| Precision |" + str(precision) + " |\n")
    file.write("| Recall |" + str(recall) + " |\n")
    file.write("| F1-Score |" + str(F1Score) + " |\n")
    file.write("| Sensitivity |" + str(sensitivity) + " |\n")
    file.write("| Specificity |" + str(specificity) + " |\n")
    file.write("\n")


 
          
plt.savefig('Confusion_Matrix.png')
file.write("![Confusion Matrix](Confusion_Matrix.png)")

file.close()

