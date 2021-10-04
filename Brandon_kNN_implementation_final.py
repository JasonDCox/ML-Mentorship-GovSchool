import math
import sys
import random
import time
import tracemalloc
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import copy

start = time.time()
tracemalloc.start()

#get file name and check if it's there
#file_name = input("Please type the name of your file, including extension: \n")
file_name = "small.arff"

try:
    input_file = open(file_name, "r")
except:
    print("File name incorrect!")
    sys.exit()

attributes = []
classes = []
data = []

looking_for = "attributes"

Lines = input_file.readlines()
for line in Lines:
    if line[0] == "@":
        stuff = line[:-1].split(" ")
        if stuff[0] == "@attribute":
            if stuff[1] != "class" and stuff[2] == "NUMERIC":
                attributes.append(stuff[1])
            elif stuff[1] == "class":
                if stuff[2] == "NUMERIC":
                    classes.append("number")
                else:
                    text = stuff[2][1:-1]
                    classes = text.split[","]
    elif line[0] == "%":
        print("",end="")
    else:
        stuff = line[:-1].split(",")
        data.append(stuff)
#print("Classes:", classes)
#print("Number of training data:", len(data))

#percentage_test = input("What percentage of the data would you like to be test data?")
percentage_test = 100
percentage_test = float(percentage_test)/100

#convert data to float numbers
final_data = []
for d in data:
    current = []
    for x in d:
        current.append(float(x))
    final_data.append(current)
data = final_data.copy()

#create test and train data
train = data.copy()
test = []
if percentage_test != 1:
    for x in range(int(len(train) * percentage_test)):
        num = random.randint(0, len(train) - 1)
        point = train[num]
        train.pop(num)
        test.append(point)

#print("length train:", len(train), "length test:", len(test), "percentage:", int((len(test)/(len(test) + len(train)))*100))

#get actual classes
classes = []
for x in data:
    c = x[-1]
    if  classes.count(c) == 0:
        classes.append(c)

#for each class, get all elements under it and add the list to dataset
dataset = []
for c in classes:
    current_class = []
    for x in train:
        if x[-1] == c:
            current = x[:-1]
            current_class.append(current)
    dataset.append(current_class)

        
def kNN (dataset, predict, group_number, k=3, m=1, p_value=1):
    global percentage_test
    #euclideam, manhattan, or minkowski
    #check if k in more than amount of classes
    if len(dataset) > k:
        #print("K is less than total classifiers!")
        print("", end="")
    temp_dataset = copy.deepcopy(dataset)
    if percentage_test == 1:
        temp_dataset[group_number].remove(predict)
    mode = m
    distances = []

    class_num = 0
    for Class in temp_dataset:
        for point in Class:
            if mode == 1:
                total_distance = 0
                for d in range(len(point)):
                    total_distance += (point[d] - predict[d]) ** 2
                euclidean_distance = math.sqrt(total_distance)
                distances.append([euclidean_distance, class_num])
            elif mode == 2:
                total_distance = 0
                for d in range(len(point)):
                    total_distance += abs(point[d] - predict[d])
                distances.append([total_distance, class_num])
            elif mode == 3:
                total_distance = 0
                for d in range(len(point)):
                    total_distance += (abs(point[d] - predict[d]))**p_value
                mikowski_distance = total_distance ** (1/p_value)
                distances.append([mikowski_distance, class_num])
            else:
                print("Invalid mode!")
        class_num += 1
    
    #sort to get closest distances and the the class of the closest three
    scored = sorted(distances)
    votes = []
    for x in scored[:k]:
        votes.append(x[1])
    
    #get most common vote
    #print(votes)
    nums = []
    groupNum = 0
    for x in range(len(dataset)):
        count = votes.count(groupNum)
        nums.append(count)
        groupNum += 1
    
    vote_result = max(votes)    
    return vote_result

#Seperate test values for different categories
test_dataset = []

if percentage_test != 1:
    for c in classes:
        current_class = []
        for x in test:
            if x[-1] == c:
                current = x[:-1]
                current_class.append(current)
        test_dataset.append(current_class)

#Create confusion matrix data
confusion_matrix = []
differences = []
squared_differences = []
corrects = 0
wrongs = 0

if percentage_test != 1:
    g = 0
    for x in test_dataset:
        class_results = [0] * len(classes)
        for y in x:
            result = kNN(dataset, y, g, 3)
            class_results[result] += 1
            if result == g:
                corrects += 1
            else:
                wrongs += 1
            differences.append(abs(result - y[-1]))
            squared_differences.append((abs(result - y[-1]))**2)
        confusion_matrix.append(class_results)
        g += 1
else:
    g = 0
    for c in dataset:
        class_results = [0] * len(classes)
        for y in c:
            result = kNN(dataset, y, g, 3)
            class_results[result] += 1
            if result == g:
                corrects += 1
            else:
                wrongs += 1
            differences.append(abs(result - y[-1]))
            squared_differences.append((abs(result - y[-1]))**2)
        confusion_matrix.append(class_results)
        #print("num:", len(dataset[g]), "results:", sum(class_results))
        g += 1

end = time.time()
total_time = end-start

horizontal_matrix = []
for i in range(len(confusion_matrix)):
    current = []
    for x in confusion_matrix:
        current.append(x[i])
    horizontal_matrix.append(current)

#create(confusion_matrix)
df_cm = pd.DataFrame(confusion_matrix, index = [i for i in classes],
                  columns = [i for i in classes])
plt.figure(figsize = (10,7))
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
sn.heatmap(df_cm, annot=True)

plt.savefig("kNN_Confusion_Matrix.png")

#create and save other data
file = open("kNN_output.txt", "w")
file.write("General Metrics: \n")
sig = "0.3f"
accuracy = corrects / (corrects + wrongs)
mean_absolute_error = sum(differences) / len(differences)
mean_squared_error = sum(squared_differences) / len(squared_differences)
file.write("Train data: " + str(len(train)) + " Test data: " + str(len(test)) + " Percentage: " + format(len(test) / (len(test) + len(train)), sig) + "\n")
file.write("Execution time: " + str(int(total_time)) + " seconds\n")
file.write("Peak Memory Usage: " + str(int(tracemalloc.get_traced_memory()[1]/1000000)) + " MB\n")
file.write("Accuracy:" + format(accuracy, sig) + "\n")
file.write("Mean absolute error:" + format(mean_absolute_error, sig) + "\n")
file.write("Mean squared error: " + format(mean_squared_error, sig) +  "\n" + "\n")
#calculate for each class:
#Precision, Recall, F1 score
class_metrics = []

group_num = 0
t = 0
for x in confusion_matrix:
    file.write("Metrics for " + str(classes[group_num]) + ":\n")
    true_posistives = x[group_num]
    false_posistives = sum(horizontal_matrix[t]) - true_posistives
    true_negatives = len(test) - (false_posistives + sum(x))
    false_negatives = sum(x) - true_posistives

    try:
        precision = format((true_posistives) / (true_posistives + false_posistives), sig)
    except:
        precision = "N/A"
    try:
        recall = format(true_posistives / (true_posistives + false_negatives), sig)
    except:
        recall = "N/A"
    try:
        f1_score = format(2 / ((1/float(recall)) * (1/float(precision))), sig)
    except:
        f1_score = "N/A"
    try:
        sensitivity = format(true_posistives / (true_posistives + false_negatives), sig)
    except:
        sensitivity = "N/A"
    try:
        specificity = format(true_negatives / (true_negatives + false_posistives), sig)
    except:
        specificity = "N/A"

    file.write("Precision: " + precision + "\n")
    file.write("Recall: " + recall + "\n")
    file.write("F1 score: " + f1_score + "\n")
    file.write("Sensitivity:" + sensitivity + "\n")
    file.write("Specificity:" + specificity + "\n")
    file.write("\n")

    group_num += 1
    t += 1


file.close()
tracemalloc.stop()
