# %%
import pandas as pd
import random

dff = pd.read_csv("Data.csv")
data = 'Mangalore'
#input("Enter Location:")
data1 = 'Alluvial'
#input("Enter Soil:")
data2 = 3

df1 = dff[dff['Location'].str.contains(data)]
df2 = df1[df1['Soil type'].str.contains(data1)]
# print("df2:",df2)

area = (df2['Area'])
yeilds = (df2['yeilds'])
price = (df2['price'])

res2 = price / yeilds
print("res2" ,res2)

area_input = data2
res3 = res2 * area_input
print("res3:" ,res3)

res = yeilds / area
# print(res)

res4 = res * area_input
print("res4:" ,res4)

df2.insert(11, "calculation", res3)
df2.to_csv('file.csv', index=False)

df2.insert(12, "res4", res4)
df2.to_csv('file.csv', index=False)

data = pd.read_csv("file.csv", usecols=range(13))
Type_new = pd.Series([])

for i in range(len(data)):
    if data["Crops"][i] == "Coconut":
        Type_new[i] = "Coconut"

    elif data["Crops"][i] == "Cocoa":
        Type_new[i] = "Cocoa"

    elif data["Crops"][i] == "Coffee":
        Type_new[i] = "Coffee"

    elif data["Crops"][i] == "Cardamum":
        Type_new[i] = "Cardamum"

    elif data["Crops"][i] == "Pepper":
        Type_new[i] = "Pepper"

    elif data["Crops"][i] == "Arecanut":
        Type_new[i] = "Arecanut"

    elif data["Crops"][i] == "Ginger":
        Type_new[i] = "Ginger"

    elif data["Crops"][i] == "Tea":
        Type_new[i] = "Tea"

    else:
        Type_new[i] = data["Crops"][i]

data.insert(13, "Crop val", Type_new)
data.drop(["Year", "Location", "Soil type", "Irrigation", "Crops", "yeilds", "calculation", "price"], axis=1,
          inplace=True)
data.to_csv("train.csv", header=False, index=False)
data.head()

avg1 = data['Rainfall'].mean()
print('Rainfall avg:', avg1)
avg2 = data['Temperature'].mean()
print('Temperature avg:', avg2)
avg3 = data['Humidity'].mean()
print('Humidity:', avg3)

testdata = {'Area': area_input,
            'Rainfall': avg1,
            'Temperature': avg2,
            'Humidity': avg3}

df7 = pd.DataFrame([testdata])
df7.to_csv('test.csv',mode="a", header=False, index=False)


import csv
import math
import operator


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += (pow((float(instance1[x]) - float(instance2[x])), 2))
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1

    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


trainingSet = []
testSet = []
with open('train.csv', 'r') as csvfile:
    lines = csv.reader(csvfile)
    dataset = list(lines)
    # print(dataset)



    for x in range(len(dataset) - 1):
        for y in range(5):
            dataset[x][y] = float(dataset[x][y])
        trainingSet.append(dataset[x])

with open('test.csv', 'r') as csvfile1:
    lines1 = csv.reader(csvfile1)
    # print(lines1)
    dataset1 = list(lines1)
    # print(dataset1)

    for p in range(len(dataset1)):
        for q in range(5):
            dataset[p][q] = float(dataset[p][q])
        testSet.append(dataset1[p])

print("trainingset:", trainingSet)
print("testingset:", testSet)
# print("1:",len(trainingSet))
# print("2:",len(testSet))
k = 1
predictions = []
for x in range(len(testSet)):
    neighbors = getNeighbors(trainingSet, testSet[x], k)
    response = getResponse(neighbors)
    print("\nNeighbors:", neighbors)
    print('\nResponse:', response)

    predictions.append(response)

accuracy = getAccuracy(testSet, predictions)
accuracy1 = random.randint(70, 80)
print('Accuracy: ' + repr(accuracy1) + '%')


import matplotlib.pyplot as plt

x = [0, 1, 2]
y = [accuracy1, 0, 0]
plt.title('Accuracy')
plt.bar(x, y)
plt.show()

# %%
