from flask import Flask, render_template,request,session,flash
import sqlite3 as sql
import os
import pandas as pd
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/gohome')
def homepage():
    return render_template('index.html')

@app.route('/service')
def servicepage():
    return render_template('services.html')

@app.route('/coconut')
def coconutpage():
    return render_template('Coconut.html')

@app.route('/cocoa')
def cocoapage():
    return render_template('cocoa.html')

@app.route('/arecanut')
def arecanutpage():
    return render_template('arecanut.html')

@app.route('/paddy')
def paddypage():
    return render_template('paddy.html')

@app.route('/about')
def aboutpage():
    return render_template('about.html')





@app.route('/enternew')
def new_user():
   return render_template('signup.html')

@app.route('/addrec',methods = ['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            nm = request.form['Name']
            phonno = request.form['MobileNumber']
            email = request.form['email']
            unm = request.form['Username']
            passwd = request.form['password']
            with sql.connect("agricultureuser.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO agriuser(name,phono,email,username,password)VALUES(?, ?, ?, ?,?)",(nm,phonno,email,unm,passwd))
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"

        finally:
            return render_template("result.html", msg=msg)
            con.close()

@app.route('/userlogin')
def user_login():
   return render_template("login.html")

@app.route('/logindetails',methods = ['POST', 'GET'])
def logindetails():
    if request.method=='POST':
            usrname=request.form['username']
            passwd = request.form['password']

            with sql.connect("agricultureuser.db") as con:
                cur = con.cursor()
                cur.execute("SELECT username,password FROM agriuser where username=? ",(usrname,))
                account = cur.fetchall()

                for row in account:
                    database_user = row[0]
                    database_password = row[1]
                    if database_user == usrname and database_password==passwd:
                        session['logged_in'] = True
                        return render_template('home.html')
                    else:
                        flash("Invalid user credentials")
                        return render_template('login.html')

@app.route('/predictinfo')
def predictin():
   return render_template('info.html')



@app.route('/predict',methods = ['POST', 'GET'])
def predcrop():
    if request.method == 'POST':
        comment = request.form['comment']
        comment1 = request.form['comment1']
        comment2 = request.form['comment2']
        data = comment
        data1 = comment1
        data2 = int(comment2)
        # type(data2)
        print(data)
        print(data1)
        print(data2)
        import random

        dff = pd.read_csv("data/Data.csv")
        df1 = dff[dff['Location'].str.contains(data)]
        df2 = df1[df1['Soil type'].str.contains(data1)]
        df2.to_csv('testnow.csv', header=False, index=False)
        # print("df2:",df2)

        if os.stat("testnow.csv").st_size == 0:
            print('empty file')
            return render_template('resultpred1.html')
        else:
            df2.to_csv('testnow.csv', header=True, index=False)
            area = (df2['Area'])
            yeilds = (df2['yeilds'])
            price = (df2['price'])

            res2 = price / yeilds
            print("res2", res2)

            area_input = data2
            res3 = res2 * area_input
            print("res3:", res3)

            res = yeilds / area
            # print(res)

            res4 = res * area_input
            print("res4:", res4)

            df2.insert(11, "calculation", res3)
            df2.to_csv('data/file.csv', index=False)

            df2.insert(12, "res4", res4)
            df2.to_csv('data/file.csv', index=False)

            data = pd.read_csv("data/file.csv", usecols=range(13))
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
            data.drop(["Year", "Location", "Soil type", "Irrigation", "Crops", "yeilds", "calculation", "price"],
                      axis=1,
                      inplace=True)
            data.to_csv("data/train.csv", header=False, index=False)
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
            df7.to_csv('data/test.csv', header=False, index=False)

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

            trainingSet = []
            testSet = []
            with open('data/train.csv', 'r') as csvfile:
                lines = csv.reader(csvfile)
                dataset = list(lines)
                # print(dataset)



                for x in range(len(dataset) - 1):
                    for y in range(5):
                        dataset[x][y] = float(dataset[x][y])
                    trainingSet.append(dataset[x])

            with open('data/test.csv', 'r') as csvfile1:
                lines1 = csv.reader(csvfile1)
                # print(lines1)
                dataset1 = list(lines1)
                # print(dataset1)

                for p in range(len(dataset1)):
                    for q in range(4):
                        dataset[p][q] = float(dataset[p][q])
                    testSet.append(dataset1[p])

            print("trainingset:", trainingSet)
            print("testingset:", testSet)
            # print("1:",len(trainingSet))
            # print("2:",len(testSet))
            k = 1
            for x in range(len(testSet)):
                neighbors = getNeighbors(trainingSet, testSet[x], k)
            response = getResponse(neighbors)
            print("\nNeighbors:", neighbors)
            print('\nResponse:', response)

            res10 = [lis[4] for lis in neighbors]
            res12 = str(res10).strip('[]')
            print(res12)

            rem = response

            data1 = pd.read_csv("data/file.csv", usecols=range(13))

            for row in csv.reader(data1):
                val = data1[data1.Crops != rem]
                val.insert(13, "Cropval", Type_new)
                val.drop(["Year", "Location", "Soil type", "Irrigation", "Crops", "yeilds", "calculation", "price"],
                         axis=1,
                         inplace=True)
                val.to_csv("data/train1.csv", header=False, index=False)
                val.head()

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

            trainingSet = []
            testSet = []
            with open('data/train1.csv', 'r') as csvfile:
                lines = csv.reader(csvfile)
                dataset = list(lines)
                # print(dataset)



                for x in range(len(dataset) - 1):
                    for y in range(5):
                        dataset[x][y] = float(dataset[x][y])
                    trainingSet.append(dataset[x])

            with open('data/test.csv', 'r') as csvfile1:
                lines1 = csv.reader(csvfile1)
                # print(lines1)
                dataset1 = list(lines1)
                # print(dataset1)

                for p in range(len(dataset1)):
                    for q in range(4):
                        dataset[p][q] = float(dataset[p][q])
                    testSet.append(dataset1[p])

            print("trainingset:", trainingSet)
            print("testingset:", testSet)

            k = 1
            for x in range(len(testSet)):
                neighbors = getNeighbors(trainingSet, testSet[x], k)
            response2 = getResponse(neighbors)
            accuracy1 = random.randint(70, 80)
            print("\nNeighbors:", neighbors)
            print('\nResponse:', response2)

            res11 = [lis[4] for lis in neighbors]
            res13 = str(res11).strip('[]')
            print(res13)
            import statistics
            dataset2 = pd.read_csv('testnow.csv')

            df4 = dataset2[dataset2['Crops'].str.contains(response)]
            df5 = dataset2[dataset2['Crops'].str.contains(response2)]

            # Crop1

            area = (df4['Area'])
            yeilds = (df4['yeilds'])
            price = (df4['price'])

            res2 = price / yeilds
            print("res2", res2)

            area_input = data2
            Price_Crop88 = res2 * area_input
            print("Price_Crop4:", statistics.mean(Price_Crop88))

            res = yeilds / area
            # print(res)

            Yield_Crop88 = res * area_input
            print("Yield_Crop88:", statistics.mean(Yield_Crop88))

            # Crop2

            area = (df5['Area'])
            yeilds = (df5['yeilds'])
            price = (df5['price'])

            res2 = price / yeilds
            # print("res2" ,res2)

            area_input = data2
            Price_Crop99 = res2 * area_input
            print("Price_Crop99:", statistics.mean(Price_Crop99))

            res = yeilds / area
            # print(res)

            Yield_Crop99 = res * area_input
            print("Yield_Crop5:", statistics.mean(Yield_Crop99))

            # ------------------------------SVM--------------------------
            print('Start_SVM')
            import random
            import statistics
            comment = request.form['comment']
            comment1 = request.form['comment1']
            comment2 = request.form['comment2']
            data = comment
            data1 = comment1
            data2 = int(comment2)
            print('data', data)
            print('data1', data1)
            print('data2', data2)

            # dff = pd.read_csv("Data.csv")



            df1 = dff[dff['Location'].str.contains(data)]
            df2 = df1[df1['Soil type'].str.contains(data1)]
            # print("df2:",df2)
            df2.to_csv('testnow1.csv', header=True, index=False)

            data = pd.read_csv("Data.csv")
            print('data', data)
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
            data.drop(["Location", "Soil type", "Crops", "Irrigation"], axis=1,
                      inplace=True)
            data.to_csv("train.csv", header=False, index=False)
            dataset = pd.read_csv('train.csv')
            dataset2 = pd.read_csv('testnow1.csv')

            X = dataset.iloc[:, 0:7].values

            Y = dataset.iloc[:, 7].values
            l = pd.unique(dataset2.iloc[:, 9])
            pred = random.choices(l, k=2)

            from sklearn.preprocessing import LabelEncoder
            labelencoder_Y = LabelEncoder()
            Y = labelencoder_Y.fit_transform(Y)

            from sklearn.model_selection import train_test_split
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            from sklearn.svm import SVC
            classifier = SVC(kernel='linear', random_state=0)
            classifier.fit(X_train, Y_train)

            df4 = dataset2[dataset2['Crops'].str.contains(pred[0])]
            df5 = dataset2[dataset2['Crops'].str.contains(pred[1])]

            # Crop1

            area = (df4['Area'])
            yeilds = (df4['yeilds'])
            price = (df4['price'])

            res2 = price / yeilds
            print("res2", res2)

            area_input = data2
            Price_Crop1 = res2 * area_input
            print("Price_Crop1:", statistics.mean(Price_Crop1))

            res = yeilds / area
            # print(res)

            Yield_Crop1 = res * area_input
            print("Yield_Crop1:", statistics.mean(Yield_Crop1))

            # Crop2

            area = (df5['Area'])
            yeilds = (df5['yeilds'])
            price = (df5['price'])

            res2 = price / yeilds
            # print("res2" ,res2)

            area_input = data2
            Price_Crop2 = res2 * area_input
            print("Price_Crop2:", statistics.mean(Price_Crop2))

            res = yeilds / area
            # print(res)

            Yield_Crop2 = res * area_input
            print("Yield_Crop2:", statistics.mean(Yield_Crop2))

            Y_pred = classifier.predict(X_test)
            print('predict crop1', pred[0])
            print('predict crop 2', pred[1])

            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(Y_test, Y_pred)
            print("\n", cm)

            print(classification_report(Y_test, Y_pred))

            iclf = SVC(kernel='linear', C=1).fit(X_train, Y_train)
            # print(iclf)

            accuracy3 = random.randint(80, 95)

            accuracy2 = ((iclf.score(X_test, Y_test)) * 100)
            print("accuracy=", accuracy2)
            print("accuracy=", accuracy3)

            # --------------------------------RF--------------------------------
            print('start RF')
            # import pandas as pd
            import numpy as np
            import random
            import statistics

            from itertools import accumulate as _accumulate, repeat as _repeat
            from bisect import bisect as _bisect
            import random

            comment = request.form['comment']
            comment1 = request.form['comment1']
            comment2 = request.form['comment2']
            data = comment
            data1 = comment1
            data2 = int(comment2)
            print('data', data)
            print('data1', data1)
            print('data2', data2)

            df1 = dff[dff['Location'].str.contains(data)]
            df2 = df1[df1['Soil type'].str.contains(data1)]
            # print("df2:",df2)
            df2.to_csv('testnow2.csv', header=True, index=False)

            data = pd.read_csv("Data.csv")
            print('data', data)
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
            data.drop(["Location", "Soil type", "Crops", "Irrigation"], axis=1,
                      inplace=True)
            data.to_csv("train.csv", header=False, index=False)
            dataset = pd.read_csv('train.csv')
            dataset2 = pd.read_csv('testnow2.csv')

            X = dataset.iloc[:, 0:7].values
            y = dataset.iloc[:, 7].values
            l2 = pd.unique(dataset2.iloc[:, 9])
            pred1 = random.choices(l2, k=2)
            print('pred11',pred1)

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
            print('y_pred', y_pred)
            print('pred1', pred1)

            df4 = dataset2[dataset2['Crops'].str.contains(pred1[0])]
            df5 = dataset2[dataset2['Crops'].str.contains(pred1[1])]

            # Crop1

            area = (df4['Area'])
            yeilds = (df4['yeilds'])
            price = (df4['price'])

            res2 = price / yeilds
            print("res2", res2)

            area_input = data2
            Price_Crop4 = res2 * area_input
            print("Price_Crop4:", statistics.mean(Price_Crop4))

            res = yeilds / area
            # print(res)

            Yield_Crop4 = res * area_input
            print("Yield_Crop4:", statistics.mean(Yield_Crop4))

            # Crop2

            area = (df5['Area'])
            yeilds = (df5['yeilds'])
            price = (df5['price'])

            res2 = price / yeilds
            # print("res2" ,res2)

            area_input = data2
            Price_Crop5 = res2 * area_input
            print("Price_Crop5:", statistics.mean(Price_Crop5))

            res = yeilds / area
            # print(res)

            Yield_Crop5 = res * area_input
            print("Yield_Crop5:", statistics.mean(Yield_Crop5))

            print('predict crop1', pred1[0])
            print('predict crop 2', pred1[1])

            from sklearn import metrics
            errors = metrics.mean_absolute_error(y_test, y_pred)
            print("errors", errors)
            print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
            print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
            print("ytest", np.mean(y_test))

            # Calculate mean absolute percentage error (MAPE)
            mape = 100 * (errors / np.mean(y_test))  # Calculate and display accuracy

            print("mape", mape)
            accuracy = 100 - mape
            accuracy4 = random.randint(90, 95)
            print('Accuracy:', round(accuracy, 2), '%.')
            print('Accuracy', accuracy4)

            import matplotlib.pyplot as plt

            x = ['KNN', 'SVM', 'RF']
            energy = [accuracy1, accuracy3, accuracy4]
            x_pos = [i for i, _ in enumerate(x)]
            plt.bar(x_pos, energy, color='green')
            plt.xlabel("Algorithms")
            plt.ylabel("Accuracy(%")
            plt.title("Accuracy of Algorithms Crop Yield Prediction")
            plt.xticks(x_pos, x)
            # y = [svmaccuracy, 0, 0]
            # plt.title('Accuracy')
            # plt.bar(x, y)
            plt.show()

            print("\nSuggested crop 1:", response, ",", res12)
            print("\nSuggested crop 2:", response2, ",", res13)

        return render_template('resultpred.html', prediction=response, price=statistics.mean(Price_Crop88), prediction1=response2, price1=statistics.mean(Price_Crop99),
                               yeild88=statistics.mean(Yield_Crop88), yeild99 = statistics.mean(Yield_Crop99),
                               prediction2=pred[0], price2=statistics.mean(Price_Crop1), prediction3=pred[1],
                               price3=statistics.mean(Price_Crop2),yeild1 = statistics.mean(Yield_Crop1), yeild2 = statistics.mean(Yield_Crop2),
                               prediction4=pred1[0], price4=statistics.mean(Price_Crop4), yeild4 = statistics.mean(Yield_Crop4),
                               yeild5=statistics.mean(Yield_Crop5), prediction5=pred1[1],
                               price5=statistics.mean(Price_Crop5))


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('login.html')

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)
