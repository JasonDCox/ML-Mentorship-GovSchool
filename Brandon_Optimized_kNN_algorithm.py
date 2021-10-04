import math
import sys
import random
import time
import tracemalloc
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import copy
# python good_kNN_algorithm.py small.arff 3 1 1
#agrs: name, k, mode, p)

print(sys.argv)
inputs = sys.argv[1:]
file_name = inputs[0]
k_value = int(inputs[1])
mode = inputs[2]

if mode == "1" or mode == "2" or mode == "3":
    mode = int(mode)
elif mode == "euclidian":
    mode = 1
elif mode == "manhattan":
    mode = 2
elif mode == "minkowski":
    mode = 3
else:
    print("Mode name invalid, going to default.")
    mode = 1

p_value = 1
try:
    p_value = float(inputs[3])
except:
    print("", end="")

start = time.time()
tracemalloc.start()

#Check if file is here
try:
    input_file = open(file_name, "r")
except:
    print("File name incorrect!")
    sys.exit()

attributes = []
classes = []
data = []

looking_for = "attributes"

print("Reading data: ", end="")

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
print("Completed")


print("Converting data: ", end="")
#convert data to float numbers
final_data = []
for d in data:
    current = []
    for x in d:
        current.append(float(x))
    final_data.append(current)
data = final_data.copy()
print("Completed")

#get actual classes
classes = []
for x in data:
    c = x[-1]
    if  classes.count(c) == 0:
        classes.append(c)
classes.sort()
print("Classes:", classes)

def most_common(lst):
    counter = 0
    num = lst[0]

    for i in lst:
        cur_count = lst.count(i)
        if (cur_count > counter):
            counter = cur_count
            num = i
    return num

predict_place = 0

def kNN (predict, k=3, p_value=1):
    global data
    global mode
    global predict_place
    #euclideam, manhattan, or minkowski
    
    distances = []
    
    if mode == 1:
        for x in data[:predict_place]:
            point = x[:-1]
            total_distance = 0
            for i in range(len(point)):
                total_distance += (point[i] - predict[i]) ** 2
            euclidean_distance = math.sqrt(total_distance)
            distances.append([euclidean_distance, x[-1]])

        for x in data[predict_place + 1:]:
            point = x[:-1]
            total_distance = 0
            for i in range(len(point)):
                total_distance += (point[i] - predict[i]) ** 2
            euclidean_distance = math.sqrt(total_distance)
            distances.append([euclidean_distance, x[-1]])
    elif mode == 2:
        for x in data[:predict_place]:
            point = x[:-1]
            total_distance = 0
            for d in range(len(point)):
                total_distance += abs(point[d] - predict[d])
            distances.append([total_distance, x[-1]])

        for x in data[predict_place + 1:]:
            point = x[:-1]
            total_distance = 0
            for d in range(len(point)):
                total_distance += abs(point[d] - predict[d])
            distances.append([total_distance, x[-1]])
    
    elif mode ==3:
        for x in data[:predict_place]:
            total_distance = 0
            for d in range(len(point)):
                total_distance += (abs(point[d] - predict[d]))**p_value
            mikowski_distance = total_distance ** (1/p_value)
            distances.append([mikowski_distance, x[-1]])

        for x in data[predict_place + 1:]:
            point = x[:-1]
            total_distance = 0
            for d in range(len(point)):
                total_distance += (abs(point[d] - predict[d]))**p_value
            mikowski_distance = total_distance ** (1/p_value)
            distances.append([mikowski_distance, x[-1]])
    else:
        print("Invalid mode!:", mode)


    #sort to get closest distances and the the class of the closest three
    scored = sorted(distances)
    votes = []
    for x in scored[:k]:
        votes.append(x[1])
    
    #get most common vote
    vote_result = most_common(votes)
    
    return vote_result

print("Starting analysis of", len(data), "data points:")

#Create confusion matrix data
confusion_matrix = []
for c in classes:
    confusion_matrix.append([0] * len(classes))
    
differences = []
squared_differences = []
corrects = 0
wrongs = 0

for d in data:
    predicted_class = classes.index(kNN(d, k_value, p_value))
    actual_class = classes.index(d[-1])

    confusion_matrix[predicted_class][actual_class] += 1

    if predicted_class == actual_class:
        corrects += 1
    else:
        wrongs += 1
    differences.append(abs(actual_class - predicted_class))
    squared_differences.append((abs(actual_class - predicted_class))**2)

    predict_place += 1

#print(len(confusion_matrix), len(confusion_matrix[0]))
print("Complete")
end = time.time()
total_time = end-start

horizontal_matrix = []
for i in range(len(confusion_matrix)):
    current = []
    for x in confusion_matrix:
        current.append(x[i])
    horizontal_matrix.append(current)
print("Exporting results: ", end="")
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
if mode == 1:
    distance_name = "euclidian"
elif mode == 2:
    distance_name = "manhattan"
elif mode == 3:
    distance_name = "minkowski"
else:
    distance_name = "unknown"
file.write("kNN analysis on: " + file_name + " with " + str(len(data)) + " data points using " + distance_name + " distance calculation\n")
file.write("General Metrics: \n")
sig = "0.3f"
accuracy = corrects / (corrects + wrongs)
mean_absolute_error = sum(differences) / len(differences)
mean_squared_error = sum(squared_differences) / len(squared_differences)
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
    true_negatives = len(data) - (false_posistives + sum(x))
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

print("Complete. Thank you for using this slow kNN algorithm!")
file.close()
tracemalloc.stop()
