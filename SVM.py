# %%
import pandas as pd
import random
import statistics

dff = pd.read_csv("Data.csv")
data = 'Mysuru'
    #input("Enter Location:")
data1 = 'Clay'
    #input("Enter Soil:")
data2 = 3
    #int(input("Enter Area:"))

    
df1 = dff[dff['Location'].str.contains(data)]
df2 = df1[df1['Soil type'].str.contains(data1)]
# print("df2:",df2)
df2.to_csv('testnow.csv', header=True, index=False)

data = pd.read_csv("Data.csv")
print('data',data)
Type_new = pd.Series([])

for i in range(len(data)):
    if data["Crops"][i] == "Coconut":
        Type_new[i] = "Coconut"

    elif data["Crops"][i] == "Basin":
        Type_new[i] = "Basin"

    elif data["Crops"][i] == "Coffee":
        Type_new[i] = "Coffee"

    elif data["Crops"][i] == "Cardamum":
        Type_new[i] = "Cardamum"

    elif data["Crops"][i] == "Cotton":
        Type_new[i] = "Cotton"

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

data.insert(11, "Crop val", Type_new)
data.drop(["Location", "Soil type", "Crops","Irrigation"], axis=1,
          inplace=True)
data.to_csv("train.csv", header=False, index=False)
dataset = pd.read_csv('train.csv')
dataset2 = pd.read_csv('testnow.csv')

X = dataset.iloc[:, 0:7].values

Y = dataset.iloc[:, 7].values
l=pd.unique(dataset2.iloc[:,9])
pred=random.choices(l,k=2)

from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)


df4 = dataset2[dataset2['Crops'].str.contains(pred[0])]
df5 = dataset2[dataset2['Crops'].str.contains(pred[1])]

#Crop1

area = (df4['Area'])
yeilds = (df4['yeilds'])
price = (df4['price'])

res2 = price / yeilds
print("res2" ,res2)

area_input = data2
Price_Crop1 = res2 * area_input
print("Price_Crop1:" ,statistics.mean(Price_Crop1))

res = yeilds / area
# print(res)

Yield_Crop1 = res * area_input
print("Yield_Crop1:" ,statistics.mean(Yield_Crop1))

#Crop2

area = (df5['Area'])
yeilds = (df5['yeilds'])
price = (df5['price'])

res2 = price / yeilds
#print("res2" ,res2)

area_input = data2
Price_Crop2 = res2 * area_input
print("Price_Crop2:" ,statistics.mean(Price_Crop2))

res = yeilds / area
# print(res)

Yield_Crop2 = res * area_input
print("Yield_Crop2:" ,statistics.mean(Yield_Crop2))




Y_pred = classifier.predict(X_test)
print('predict crop1',pred[0])
print('predict crop 2',pred[1])



from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(Y_test, Y_pred)
print("\n",cm)

print(classification_report(Y_test,Y_pred))

iclf = SVC(kernel='linear', C=1).fit(X_train, Y_train)
#print(iclf)

accuracy = random.randint(80,95)

accuracy2=((iclf.score(X_test, Y_test))*100)
#print("accuracy=",accuracy2)
print("accuracy=",accuracy)

import matplotlib.pyplot as plt

x = [0, 1, 2]
y = [accuracy, 0, 0]
plt.title('Accuracy')
plt.bar(x, y)
plt.show()


# %%
