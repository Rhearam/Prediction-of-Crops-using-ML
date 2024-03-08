# %%


# %%
import pandas as pd
import numpy as np
import random

from itertools import accumulate as _accumulate, repeat as _repeat
from bisect import bisect as _bisect
import random
def choices(population, weights=None, *, cum_weights=None, k):
    """Return a k sized list of population elements chosen with replacement.
    If the relative weights or cumulative weights are not specified,
    the selections are made with equal probability.
    """
    n = len(population)
    if cum_weights is None:
        if weights is None:
            _int = int
            n += 0.0    # convert to float for a small speed improvement
            return [population[_int(random.random() * n)] for i in _repeat(None, k)]
        cum_weights = list(_accumulate(weights))
    elif weights is not None:
        raise TypeError('Cannot specify both weights and cumulative weights')
    if len(cum_weights) != n:
        raise ValueError('The number of weights does not match the population')
    bisect = _bisect
    total = cum_weights[-1] + 0.0   # convert to float
    hi = n - 1
    return [population[bisect(cum_weights, random.random() * total, 0, hi)]
            for i in _repeat(None, k)]

dff = pd.read_csv("Data.csv")
data = 'Mangalore'
    #input("Enter Location:")
data1 = 'Alluvial'
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
X = dataset.iloc[:, 0:7].values
y = dataset.iloc[:, 7].values
dataset2 = pd.read_csv('testnow.csv')
l=pd.unique(dataset2.iloc[:,9])
pred=random.choices(l,k=2)

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('y_pred',y_pred)
print('pred',pred)
from sklearn import metrics
errors=metrics.mean_absolute_error(y_test, y_pred)
print("errors",errors)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("ytest",np.mean(y_test))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / np.mean(y_test))# Calculate and display accuracy

print("mape",mape)
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')

import matplotlib.pyplot as plt

x = [0, 1, 2]
y = [accuracy, 0, 0]
plt.title('Accuracy')
plt.bar(x, y)
plt.show()



# %%
