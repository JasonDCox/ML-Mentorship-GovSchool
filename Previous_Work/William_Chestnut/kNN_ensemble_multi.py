import math

def euclidian(point1, point2):
    total_distance = 0
    for i in range(len(point1)-1):
        total_distance += (float(point1[i]) - float(point2[i])) ** 2
    return math.sqrt(total_distance)
def manhattan(point1, point2):
    total_distance = 0
    for d in range(len(point1) - 2):
        total_distance += abs(float(point1[d]) - float(point2[d]))
    return total_distance

def minkowski(point1, point2, p):
    total_distance = 0
    for i in range(len(point1) - 2):
        total_distance += (abs(float(point1[i]) - float(point2[i])))**p
    return total_distance ** (1/p)

def majority(lst):
    counter = 0
    num = lst[0]

    for i in lst:
        cur_count = lst.count(i)
        if (cur_count > counter):
            counter = cur_count
            num = i
    return num

def kNN(member):
    start, finish, file_name, k_values, modes, p_value = member
    batch_guesses = []

    try:
        input_file = open(file_name, "r")
    except:
        print("File name incorrect!")
        sys.exit()

    Lines = input_file.readlines()
    data = []
    for l in Lines:
        if l[0] != "@":
            line = l.strip("\n").split(",")
            data.append(line)
    
    for predict in data[start:finish]:
        ensemble_votes = []
        for k in k_values:
            for m in modes:
                distances = [9999999999999999] * k
                classes = [9999999999999999] * k
                highest = max(distances)
                taken = False
                for point in data:
                    if point != predict or taken == True:
                        if m == 1:
                            distance = euclidian(predict, point)
                        elif m == 2:
                            distance = manhattan(predict, point)
                        elif m == 3:
                            distance = minkowski(predict, point, p_value)
                        else:
                            print("Invalid mode!")
                            sys.exit()

                        if distance < highest:
                            classes.pop(distances.index(highest))
                            distances.remove(highest)
                            distances.append(distance)
                            classes.append(point[-1])
                            highest = max(distances)
                    else:
                        taken = True
                ensemble_votes.append(majority(classes))
        guess = majority(ensemble_votes)
        batch_guesses.append(guess)

    print("done", end=" ")
    return batch_guesses
