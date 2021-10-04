from scipy.io import arff
import pandas as pd
import numpy as np

def most_frequent(lst):
    count0 = lst.count(0.0)
    count1 = lst.count(1.0)
    count2 = lst.count(2.0)
    count3 = lst.count(3.0)
    count4 = lst.count(4.0)
    count5 = lst.count(5.0)
    count6 = lst.count(6.0)
    count7 = lst.count(7.0)
    counts = (count0, count1, count2, count3, count4, count5, count6, count7)
    frequent_ind = counts.index(max(counts))
    return(frequent_ind)
    


data = arff.loadarff("C:\\Users\\Jason\\Documents\\Mentorship\\small.arff") 

df = pd.DataFrame(data[0])

target = df['class']

df = df.drop(['class'], axis=1)

X = df.values
Y = target.values 

ind = np.arange(0,len(X))
np.random.shuffle(ind)

Xs = X[ind]
Ys = Y[ind]

testNum = int(np.floor(0.1 * len(X)))
X_test = Xs[:testNum]
Y_test = Ys[:testNum]
Xs = Xs[testNum :]
Ys = Ys[testNum :]

for i in X_test:
    euc_dis = []
    for j in Xs: 
        euc_dis.append(((i[0] - j[0])**2 + (i[1] - j[1])**2 + (i[2] - j[2])**2 + (i[3] - j[3])**2 + (i[4] - j[4])**2 + (i[5] - j[5])**2 + (i[6] - j[6])**2))
        index_sort_dis = np.argsort(euc_dis)
    closest_ind = index_sort_dis[:3]
    closest = []
    for x in closest_ind:
        closest.append(Ys[x])
    f = most_frequent(closest)
    if f == 0:
        print(i, " = 0")
    elif f  == 1:
        print(i, " = 1.0")
    elif f  == 2:
        print(i, " = 2.0")
    elif f  == 3:
        print(i, " = 3.0")
    elif f  == 4:
        print(i, " = 4.0")
    elif f  == 5:
        print(i, " = 5.0")
    elif f  == 6:
        print(i, " = 6.0")
    elif f  == 7:
        print(i, " = 7.0")
    else: 
        print("Failed")
